import torch
from torch import nn
from socket import socket
import time
import os
import threading
from config_run import *
from communicator import Communicator, Message
from models import BaseModel, get_model_by_name
from aggregator import Aggregator
from utils import *
from load_data import get_test_dataloader


class Server(Communicator, Aggregator):
    def __init__(
            self, 
            ip_address: str, 
            port: int, 
            num_training_rounds: int, 
            learning_rate: float, 
            momentum: float, 
            model_name: str,
            ) -> None:
        super(Server, self).__init__()
        self.ip = ip_address
        self.port = port
        self.num_training_rounds = num_training_rounds
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model = get_model_by_name(model_name)
        self.best_model = self.model

        # One for each client: dict[client_ip: value]
        self.client_socks = {}
        self.partition_points = {}
        self.server_models = {} 
        self.optimizers = {}
        self.loss_fns = {}

        self.current_training_round = 0
        self.training_times = {}
        self.testing_loss_fn = nn.CrossEntropyLoss()
        self.test_dataloader = get_test_dataloader(DATASET, 500)
        self.test_accuracy = []
        self.test_average_loss = []
        self.lock = threading.Lock()
        self.measurements = []
        self.server_round_measurements = {}
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


        self.sock.bind(("", self.port))
        self._start_server()

    def _start_server(self) -> None:
        while len(self.client_socks) < NUM_CLIENTS:
            print(f"[SERVER] Listening at {self.ip}:{self.port}")
            self.sock.listen()
            conn, (ip, port) = self.sock.accept()
            # Send training info to client
            self._send_training_info(conn)
            client_info = self.recv_msg(conn, "MSG_CLIENT_INFO")
            client_ip = f"{ip}:{port}"
            self.partition_points[client_ip] = client_info.body["partition_point"]
            self.client_socks[client_ip] = conn
            client_registration = len(self.client_socks) - 1

            self.send_msg(conn, Message("MSG_REGISTERED", client_registration))
            print(f"[SERVER] {len(self.client_socks)}/{NUM_CLIENTS} clients connected.")

    def _send_training_info(self, conn: socket) -> None:
        self.send_msg(conn, Message(
            "MSG_TRAINING_INFO", 
            {
                "num_rounds": self.num_training_rounds,
                "learning_rate": self.learning_rate,
                "momentum": self.momentum
            }
        ))

    def send_models(self, client_models: dict[str, BaseModel] = None) -> None:
        if client_models is None:
            client_models = self._initialize_models()
        
        for client_ip in self.client_socks:
            client_sock = self.client_socks[client_ip]
            client_model = client_models[client_ip]
            
            self.send_msg(client_sock, Message("MSG_CLIENT_MODEL", {
                "model": client_model,
                "train_full_model": self._is_full_model(client_ip)
            }))

    def train_round(self, round: int) -> None:
        print(f"[SERVER] Start training round {round}")
        self.current_training_round = round
        self.average_losses_round = {}
        self.train_accuracy_round = {}
        threads = {}
        for client_ip in self.client_socks:
            if self._is_full_model(client_ip):
                continue # Trains fully on client
            threads[client_ip] = threading.Thread(target=self._train_round_client, args=(client_ip,))
        
        start_train = time.time()
        for thread in threads.values():
            thread.start()

        for thread in threads.values():
            thread.join()
        end_train = time.time()

        # Save server round measurements
        self.server_round_measurements = {
            "round": self.current_training_round,
            "id": f"{self.ip}:{self.port}",
            "training_start": start_train,
            "training_end": end_train,
            "training_time": self.training_times,
            "partition_point": None,
            "average_loss": None,   
            "train_accuracy": None
        }

    def _train_round_client(self, client_ip: str) -> None:
        total_loss = 0
        correct = 0
        total_samples = 0
        self.training_times[client_ip] = 0
        client_sock = self.client_socks[client_ip]
        num_local_rounds = 0

        while True:
            intermediate_data_msg = self.recv_msg(client_sock, "MSG_INTERMEDIATE_DATA")
            intermediate_data = intermediate_data_msg.body
            activations, labels, is_last_round = intermediate_data.values()

            activations, labels = activations.to(self.device), labels.to(self.device)

            # Lock thread for autograd thread safety
            with self.lock:
                start_train = time.time()
                self.optimizers[client_ip].zero_grad()
                server_model = self.server_models[client_ip].to(self.device)
                outputs = server_model(activations)

                loss = self.loss_fns[client_ip](outputs, labels)
                activations.retain_grad()
                loss.backward()
                self.optimizers[client_ip].step()
                
                gradients = activations.grad
                gradients = gradients.to("cpu")
                
                total_loss += loss.item()
                _, pred = outputs.max(1)
                total_samples += labels.size(0)
                correct += pred.eq(labels).sum().item()

                self.training_times[client_ip] += (time.time() - start_train)
            
            self.send_msg(client_sock, Message("MSG_GRADIENTS", gradients))
            
            num_local_rounds += 1
            if is_last_round:
                break
            
        average_loss = total_loss/num_local_rounds
        accuracy = 100. * correct/total_samples
        self.train_accuracy_round[client_ip] = accuracy
        self.average_losses_round[client_ip] = average_loss
        print(f"[SERVER] Training accuracy for client {client_ip}: {accuracy}")
        print(f"[SERVER] Average loss for client {client_ip}: {average_loss}")

    def aggregate_round(self) -> None:
        # Receive and aggregate client models
        print("[SERVER] Collecting client models")
        combined_models = {}
        for client_ip in self.client_socks:
            round_measurements = {}
            client_sock = self.client_socks[client_ip]

            client_model_msg = self.recv_msg(client_sock, "MSG_CLIENT_MODEL_FINISHED")
            client_model = client_model_msg.body
            
            client_info_msg = self.recv_msg(client_sock, "MSG_CLIENT_ROUND_INFO")
            client_info = client_info_msg.body

            if self._is_full_model(client_ip):
                combined_models[client_ip] = client_model.to("cpu")
            else:
                combined_models[client_ip] = self._create_global_model(
                    client_model=client_model.to("cpu"),
                    server_model=self.server_models[client_ip].to("cpu")
                )

            # Save client round measurements
            round_measurements["round"] = self.current_training_round
            round_measurements["id"] = client_ip
            for key in client_info.keys():
                if key == "average_loss":
                    self.average_losses_round[client_ip] = client_info[key]
                    continue
                if key == "train_accuracy":
                    self.train_accuracy_round[client_ip] = client_info[key]
                    continue
                round_measurements[key] = client_info[key]
            round_measurements["partition_point"] = self.partition_points[client_ip]
            round_measurements["average_loss"] = self.average_losses_round[client_ip]
            round_measurements["train_accuracy"] = self.train_accuracy_round[client_ip]
            self.measurements.append(round_measurements)

        if NUM_CLIENTS > 1:
            print("[SERVER] Aggregating models")
            aggregated_models = self._aggregate_models(combined_models)
            print("[SERVER] Updating global model")
            self.model = aggregated_models[client_ip] # Models are identical
        else:
            print("[SERVER] Updating global model")
            self.model = combined_models[client_ip]

        self.measurements.append(self.server_round_measurements)

        if self.current_training_round < (self.num_training_rounds - 1):
            # Partition models again - could also be a different partition point in the future
            client_models = self._initialize_models()
            self.send_models(client_models)
            
            os.makedirs(SAVE_PATH_MODELS, exist_ok=True)

            if self.current_training_round % 5 == 0:
                torch.save(self.model.state_dict(), f"{SAVE_PATH_CHECKPOINT}{self.current_training_round}.pt")
                print("[SERVER] Checkpoint model saved")
        else:
            os.makedirs(SAVE_PATH_MODELS, exist_ok=True)
            os.makedirs(RESULTS_PATH, exist_ok=True)
            save_measurements_to_csv(self.measurements, MEASUREMENTS_PATH)
            torch.save(self.best_model.state_dict(), f"{SAVE_PATH_MODELS}{MODEL}_best.pt")
            print("[SERVER] Best model saved")

    def test(self) -> None:
        print("[SERVER] Evaluating global model")
        self.model = self.model.to(self.device)
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.testing_loss_fn(outputs, labels)
                loss = loss.detach()
                total_loss += loss.item()
                _, pred = outputs.max(1)
                total_samples += labels.size(0)
                correct += pred.eq(labels).sum().item()

        average_loss = total_loss/len(self.test_dataloader)
        accuracy = 100. * correct/total_samples
        self.test_average_loss.append(average_loss)
        self.test_accuracy.append(accuracy)
        self.model = self.model.to("cpu")

        print(f"[SERVER] Test accuracy: {accuracy}, loss: {average_loss}")

        if accuracy > max(self.test_accuracy):
            print("[SERVER] Updating best model")
            self.best_model = self.model

    def disconnect(self) -> None:
        self.sock.close()
        print("[SERVER] Disconnected")

    def _partition_model(self, client_ip: str) -> tuple[BaseModel, BaseModel]:
        # Partition point is the last nn.ModuleList layer executed on the client
        if client_ip not in self.partition_points:
            print(f"[SERVER] No partition point provided for client {client_ip}")
            self.partition_points[client_ip] = self._get_optimal_partition_point(client_ip)
            print(f"[SERVER] Set partition point to '{self.partition_points[client_ip]}' for client {client_ip}")
        elif not self._is_valid_partition_point(self.partition_points[client_ip]):
            print(f"[SERVER] '{self.partition_points[client_ip]}' is not a valid partition point for client {client_ip}")
            self.partition_points[client_ip] = self._get_optimal_partition_point(client_ip)
            print(f"[SERVER] Set partition point to '{self.partition_points[client_ip]}' for client {client_ip}")

        partition_point = self.partition_points[client_ip]
        client_model = BaseModel()
        client_model.set_layers_client(self.model.layers, partition_point)
        server_model = BaseModel()
        server_model.set_layers_server(self.model.layers, partition_point)
        return client_model, server_model
    
    def _initialize_models(self) -> dict[str, BaseModel]:
        print("[SERVER] Initializing models")
        client_models = {}

        for client_ip in self.client_socks.keys():
            client_model, server_model = self._partition_model(client_ip)
            client_models[client_ip] = client_model
            self.server_models[client_ip] = server_model
            if not self._is_full_model(client_ip):
                self.optimizers[client_ip] = torch.optim.SGD(
                    server_model.parameters(), 
                    lr=self.learning_rate, 
                    momentum=self.momentum
                    )
                self.loss_fns[client_ip] = nn.CrossEntropyLoss()
        return client_models
    
    def _get_optimal_partition_point(self, client_ip: str) -> int:
        # TODO: Derive ideal partition points based on optimization criteria
        # Set to 0 instead for now
        return 0
    
    def _is_valid_partition_point(self, partition_point: int) -> bool:
        if partition_point is None:
            return False
        return (partition_point >= 0) and \
        (partition_point <= (len(self.model.layers)-1))
    
    def _is_full_model(self, client_ip: str) -> bool:
        return self.partition_points[client_ip] == (len(self.model.layers)-1)

    def _aggregate_models(self, models: dict[str, BaseModel]) -> dict[str, BaseModel]:
        updated_weights = self.fed_avg(models)
        for model in models.values():
            self.update_model(model, updated_weights)
        return models
    
    def _create_global_model(self, client_model: BaseModel, server_model: BaseModel) -> BaseModel:
        global_model = BaseModel()
        global_model.set_layers(client_model.layers)
        global_model.set_layers(server_model.layers)
        return global_model
