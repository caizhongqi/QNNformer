import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4  # 量子比特的数量
n_layers = 2  # 量子层的深度

dev = qml.device('default.qubit', wires=n_qubits)

def enhanced_quantum_circuit(inputs, weights):
    """改进的量子电路，增加表达能力"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # 采用更多的纠缠操作
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CZ(wires=[n_qubits - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, input_features):
        super(QuantumLayer, self).__init__()
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features}) is not divisible by split size ({self.split_size}).")

        self.qnodes = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(enhanced_quantum_circuit, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        # 改进后的经典融合层，增加多层感知机结构
        self.fc = nn.Sequential(
            nn.Linear(n_qubits * self.n_splits, input_features),
            nn.ReLU(),
            nn.Linear(input_features, input_features)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"features size {features} 不匹配 {self.split_size * self.n_splits}"

        x_splits = torch.chunk(x, self.n_splits, dim=-1)
        q_outputs = []

        for i in range(self.n_splits):
            x_split = x_splits[i].reshape(-1, self.split_size)
            q_output = self.qnodes[i](x_split)
            q_outputs.append(q_output)

        # 将处理后的分块重新拼接
        x = torch.cat(q_outputs, dim=-1)
        x = self.fc(x)  # 通过改进后的全连接层进行融合
        x = x.reshape(batch_size, seq_len, -1)
        return x
