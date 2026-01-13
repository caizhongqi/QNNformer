import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 6
n_layers = 3

dev = qml.device('default.qubit', wires=n_qubits)

def prepare_entangled_state(wires):
    """Creates an entangled GHZ state across the given qubits."""
    qml.Hadamard(wires=wires[0])
    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[0], wires[i]])

def apply_qft(wires):
    """Applies Quantum Fourier Transform on the given qubits."""
    for i in range(len(wires)):
        qml.Hadamard(wires=wires[i])
        for j in range(i + 1, len(wires)):
            qml.CRZ(np.pi / (2 ** (j - i)), wires=[wires[j], wires[i]])
    for i in range(len(wires) // 2):
        qml.SWAP(wires=[wires[i], wires[-i - 1]])

def apply_inverse_qft(wires):
    """Applies Inverse Quantum Fourier Transform on the given qubits."""
    for i in range(len(wires) // 2):
        qml.SWAP(wires=[wires[i], wires[-i - 1]])
    for i in reversed(range(len(wires))):
        for j in reversed(range(i + 1, len(wires))):
            qml.CRZ(-np.pi / (2 ** (j - i)), wires=[wires[j], wires[i]])
        qml.Hadamard(wires=wires[i])

def quantum_circuit(inputs, weights):
    # 使用复杂的量子态准备方法
    prepare_entangled_state(wires=range(n_qubits))
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    # 应用量子傅里叶变换 (QFT)
    apply_qft(wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.U3(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

        # 使用复杂的纠缠结构
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, (i + 1) % n_qubits])
        qml.CZ(wires=[n_qubits - 1, 0])

    # 应用逆量子傅里叶变换 (IQFT)
    apply_inverse_qft(wires=range(n_qubits))

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
                qml.QNode(quantum_circuit, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size, 3)}
            ) for _ in range(self.n_splits)
        ])

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"features size {features} 不匹配 {self.split_size * self.n_splits}"

        x_splits = torch.chunk(x, self.n_splits, dim=-1)
        q_outputs = []

        for i in range(self.n_splits):
            x_split = x_splits[i].reshape(-1, self.split_size)
            q_output = self.qnodes[i](x_split)
            q_outputs.append(q_output)

        x = torch.cat(q_outputs, dim=-1)
        x = x.reshape(batch_size, seq_len, -1)
        return x
