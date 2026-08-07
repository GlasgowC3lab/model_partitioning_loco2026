import socket
import pickle
import struct
import time

class Message():
    def __init__(self, type: str, body: any =None) -> None:
        self.type = type
        self.body = body

class Communicator():
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def send_msg(self, conn: socket.socket, msg: Message) -> None:
        msg_bytes = pickle.dumps(msg)
        conn.sendall(struct.pack(">I", len(msg_bytes)))
        conn.sendall(msg_bytes)

    def recv_msg(self, conn: socket.socket, expected_type: str) -> Message:
        msg_len = struct.unpack(">I", conn.recv(4))[0]
        msg_bytes = conn.recv(msg_len, socket.MSG_WAITALL)
        msg = pickle.loads(msg_bytes)
        if expected_type != msg.type:
            raise Exception(f"Actual type {msg.type} does not match expected type {expected_type}")
        return msg
       