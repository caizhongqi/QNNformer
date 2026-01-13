import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4  # 量子比特的数量
n_layers = 3  # 量子层的深度

# 定义量子设备
dev = qml.device('default.qubit', wires=n_qubits)

def quantum_tunneling_effect(wires, tunneling_weights):
    """使用多种可学习参数的量子门模拟量子隧穿效应"""
    for i in range(len(wires) - 1):
        # 增加更多种类的量子门
        qml.CRX(tunneling_weights[i], wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CZ(wires=[wires[i], wires[i + 1]])
    qml.CRX(tunneling_weights[-1], wires=[wires[-1], wires[0]])
    qml.CNOT(wires=[wires[-1], wires[0]])
    qml.CZ(wires=[wires[-1], wires[0]])
    qml.SWAP(wires=[wires[0], wires[1]])  # 增加SWAP门

def quantum_circuit_with_tunneling(inputs, weights, tunneling_weights, measurement_weights):
    """使用增强的量子隧穿效应的量子电路，包含更多种类的量子测量"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # 应用增强的量子隧穿效应，并使用可学习的参数
        quantum_tunneling_effect(range(n_qubits), tunneling_weights[layer])

    # 使用可学习的参数对量子态进行不同的测量操作
    return [qml.expval(qml.PauliZ(i) * measurement_weights[0]) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i) * measurement_weights[1]) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliY(i) * measurement_weights[2]) for i in range(n_qubits)]

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
                qml.QNode(quantum_circuit_with_tunneling, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size),
                 "tunneling_weights": (n_layers, self.split_size),
                 "measurement_weights": (3,)}  # 增加可学习的测量权重
            ) for _ in range(self.n_splits)
        ])

        # 初始化量子节点的权重
        for qnode in self.qnodes:
            qnode.weights.data = torch.tensor(np.random.uniform(low=-0.01, high=0.01, size=qnode.weights.shape))

        # 增强后的全连接层，增加多层感知机结构，并处理扩展后的量子测量结果
        self.fc = nn.Sequential(
            nn.Linear(3 * n_qubits * self.n_splits, 256),  # 扩展隐藏层维度
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),  # 增加多层隐藏层
            nn.Linear(128, input_features),
            nn.ReLU()
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

        # 通过增强后的全连接层融合量子特征
        x = self.fc(x)

        # 恢复形状为 [batch_size, seq_len, -1]
        x = x.reshape(batch_size, seq_len, -1)
        return x
