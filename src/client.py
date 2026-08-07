from communicator import Communicator, Message
from load_data import get_client_dataloader, get_training_data
import torch
from torch import nn
import time
from config_run import DATASET

class Client(Communicator):
    def __init__(
            self, 
            server_ip: str, 
            server_port: int,
            partition_point: int = None
            ) -> None:
        super(Client, self).__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.registered = False
        self.model = None
        self.current_training_round = 0
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.training_time = []
        self.partition_point = partition_point
        
        self._register_client()
        
        while self.registered:
            self._initialize_round()

            if self.train_full_model:
                self._train_round_full_model()
            else:
                self._train_round()

            if self.current_training_round == self.num_training_rounds:
                self._unregister_client()

    def _register_client(self) -> None:
        self.sock.connect((self.server_ip, self.server_port))
        training_info_msg = self.recv_msg(self.sock, "MSG_TRAINING_INFO")
        training_info = training_info_msg.body
        self.num_training_rounds = training_info["num_rounds"]
        self.learning_rate = training_info["learning_rate"]
        self.momentum = training_info["momentum"]

        self.send_msg(self.sock, Message("MSG_CLIENT_INFO", {
            "partition_point": self.partition_point,
        }))

        # Receive "MSG_REGISTERED" after successful registration
        response = self.recv_msg(self.sock, "MSG_REGISTERED")
        self.client_id = response.body # Registration number used to partition dataset
        self.registered = True
        print(f"[CLIENT {self.client_id}] Registered with server at {self.server_ip}:{self.server_port}")

    def _unregister_client(self) -> None:
        self.registered = False
        self.sock.close()
        print(f"[CLIENT {self.client_id}] Disconnected")

    def _initialize_round(self) -> None:
        model_msg = self.recv_msg(self.sock, "MSG_CLIENT_MODEL")
        self.model = model_msg.body["model"]
        self.train_full_model = model_msg.body["train_full_model"]
        print(f"[CLIENT {self.client_id}] Start training round {self.current_training_round}")
        
        if self.current_training_round == 0:
            training_data = get_training_data(DATASET)
            self.train_dataloader = get_client_dataloader(training_data, self.client_id)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _train_round(self) -> None:
        training_time = 0
        round_counter = 0

        self.model = self.model.to(self.device)
        self.model.train()
        start_train_timestamp = time.time()
        for batch, (inputs, labels) in enumerate(self.train_dataloader):
            start_train = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            activations = self.model(inputs)

            # "Pause" computation timer until training resumes
            training_time += (time.time() - start_train)
            round_counter += 1
            
            activations_cpu, labels_cpu = activations.to("cpu"), labels.to("cpu")
            intermediate_data = {
                "activations": activations_cpu,
                "labels": labels_cpu,
                "is_last_round": (round_counter == len(self.train_dataloader))
            }

            self.send_msg(self.sock, Message("MSG_INTERMEDIATE_DATA", intermediate_data))
            gradients_msg = self.recv_msg(self.sock, "MSG_GRADIENTS")
            gradients = gradients_msg.body
            
            gradients = gradients.to(self.device)
            resume_train = time.time()
            activations.backward(gradients)
            self.optimizer.step()
            training_time += (time.time() - resume_train)
            
        end_train = time.time()
        print(f"[CLIENT {self.client_id}] Finished training round {self.current_training_round}")
        self.training_time.append(training_time)

        # Send model and round measurements to server
        self.model = self.model.to("cpu")
        self.send_msg(self.sock, Message("MSG_CLIENT_MODEL_FINISHED", self.model))
        msg = {
            "training_start": start_train_timestamp,
            "training_end": end_train,
            "training_time": training_time
        }
        self.send_msg(self.sock, Message("MSG_CLIENT_ROUND_INFO", msg))
        
        self.current_training_round += 1

    def _train_round_full_model(self) -> None:
        training_time = 0
        total_loss = 0
        correct = 0
        total_samples = 0

        start_train = time.time()
        self.model = self.model.to(self.device)
        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total_samples += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
        end_train = time.time()
        print(f"[CLIENT {self.client_id}] Finished training round {self.current_training_round}")
        accuracy = 100. * correct/total_samples
        training_time += (end_train - start_train)
        self.training_time.append(training_time)
        average_loss = total_loss/len(self.train_dataloader)
        print(f"[CLIENT {self.client_id}] Training accuracy: {accuracy}")
        print(f"[CLIENT {self.client_id}] Average loss: {average_loss}")

        # Send model and round measurements to server
        self.model = self.model.to("cpu")
        self.send_msg(self.sock, Message("MSG_CLIENT_MODEL_FINISHED", self.model))
        msg = {
            "training_start": start_train,
            "training_end": end_train,
            "training_time": training_time,
            "average_loss": average_loss,
            "train_accuracy": accuracy
        }
        self.send_msg(self.sock, Message("MSG_CLIENT_ROUND_INFO", msg))

        self.current_training_round += 1
