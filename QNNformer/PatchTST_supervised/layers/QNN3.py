import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4  # 量子比特的数量
n_layers = 2  # 量子层的深度

# 定义量子设备
dev = qml.device('default.qubit', wires=n_qubits)

def enhanced_quantum_circuit(inputs, weights):
    """改进的量子电路，增加表达能力"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # 增强纠缠操作
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CZ(wires=[n_qubits - 1, 0])

    # 返回多个测量结果以捕捉不同的特征
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

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
                qml.QNode(enhanced_quantum_circuit, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        # 全连接层，增加多层感知机结构，并处理扩展后的量子测量结果
        self.fc = nn.Sequential(
            nn.Linear(2 * n_qubits * self.n_splits, input_features),  # 注意：输入维度是 2 倍的 n_qubits
            nn.ReLU(),
            nn.Linear(input_features, input_features)
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

        # 通过全连接层融合量子特征
        x = self.fc(x)

        # 恢复形状为 [batch_size, seq_len, -1]
        x = x.reshape(batch_size, seq_len, -1)
        return x
