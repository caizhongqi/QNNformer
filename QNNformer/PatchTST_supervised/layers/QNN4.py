import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4
n_layers = 2  # 使用较浅的层数

# 定义量子设备
dev = qml.device('default.qubit', wires=n_qubits)

def mera_layer(wires):
    """实现一个简化的 MERA 层"""
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.SWAP(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[3]])

def quantum_error_correction(wires):
    """模拟简单的量子纠错编码和解码"""
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.CNOT(wires=[wires[1], wires[3]])  # 模拟一个纠错码

def quantum_circuit_with_mera_and_qec(inputs, weights):
    """结合 MERA 和 QEC 的量子电路"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # 应用 MERA 结构
        mera_layer(range(n_qubits))

        # 应用量子纠错
        quantum_error_correction(range(n_qubits))

    # 测量不同的 Pauli 操作
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]  # 增加Pauli-Y的测量

class QuantumLayer(nn.Module):
    def __init__(self, input_features):
        super(QuantumLayer, self).__init__()
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features}) cannot be divided evenly by split size ({self.split_size}).")

        # 创建多个量子节点来处理输入的不同分块
        self.qnodes = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_mera_and_qec, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        # 添加层归一化以稳定训练过程
        self.norm = nn.LayerNorm(3 * n_qubits * self.n_splits)

        # 经典全连接层，增强数据特征提取能力
        self.fc = nn.Sequential(
            nn.Linear(3 * n_qubits * self.n_splits, 64),  # 输出维度为 64
            Swish(),  # 使用 Swish 激活函数
            nn.Linear(64, input_features)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"Features size {features} does not match {self.split_size * self.n_splits}"

        # 处理每一个分块，并通过量子层
        q_outputs = []
        for i in range(self.n_splits):
            x_split = x[:, :, i * self.split_size:(i + 1) * self.split_size]
            x_split = x_split.reshape(-1, self.split_size)
            q_output = self.qnodes[i](x_split)
            q_outputs.append(q_output)

        # 将量子层输出拼接起来
        x = torch.cat(q_outputs, dim=-1)

        # 应用层归一化
        x = self.norm(x)

        # 通过经典全连接层融合量子特征
        x = self.fc(x)

        # 恢复形状为 [batch_size, seq_len, -1]
        x = x.reshape(batch_size, seq_len, -1)
        return x

# 自定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
