# 用于模拟客户端发送Protobuf消息到server，不部署在我们的系统中
# -*- coding: utf-8 -*-
import time

import Pose_pb2
import socket
import random

def data():
    # 1. 创建Protobuf消息对象并填充数据
    my_msg = Pose_pb2.MainCarInfo()
    my_msg.x = 7987.086396549613
    my_msg.y = -2964.653905917249
    my_msg.z = -396.08292995686327
    my_msg.yaw = 1.9627021134112075
    my_msg.roll = 0
    my_msg.pitch = 0
    print(my_msg.x, my_msg.y)
    return my_msg

class TCPClient:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_message_pb(self):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            # 2. 序列化为字节串
            my_msg = data()
            serialized_data = my_msg.SerializeToString()
            print(serialized_data)
            # 3. 计算长度并打包为4字节的消息头 (使用大端字节序)
            data_length = len(serialized_data)
            length_header = data_length.to_bytes(4, byteorder='big')  # 4字节头
            # 先发送长度头，再发送序列化数据
            self.socket.sendall(length_header + serialized_data)
            #self.socket.send(message.encode('utf-8'))

            #time.sleep(100)
            # 接收响应
            response = self.socket.recv(1024)
            print(f"服务器响应: {response.decode('utf-8')}")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def send_message(self, message):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            self.socket.send(message.encode('utf-8'))

            # 接收响应
            response = self.socket.recv(1024)
            print(f"服务器响应: {response.decode('utf-8')}")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            print("连接已关闭")

if __name__ == "__main__":
    client = TCPClient()

    if client.connect():
        # 发送测试消息
        messages = ["Hello, Server!", "测试消息2", "测试消息3"]

        for msg in range(1,11):
            #client.send_message(msg)
            client.send_message_pb()

        client.close()

    