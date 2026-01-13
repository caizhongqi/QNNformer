import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4  # 减少量子比特的数量
n_layers = 1  # 减少量子层的深度

dev = qml.device('default.qubit', wires=n_qubits)

def simple_quantum_circuit(inputs, weights):
    # 简化量子态准备方法，仅使用Hadamard态
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)

        # 仅使用简单的纠缠结构
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

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
                qml.QNode(simple_quantum_circuit, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        # 加入经典的全连接层，用于融合量子特征
        self.fc = nn.Linear(n_qubits * self.n_splits, input_features)

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
        x = self.fc(x)  # 融合量子特征
        x = x.reshape(batch_size, seq_len, -1)
        return x
