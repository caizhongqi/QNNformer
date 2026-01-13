import torch
import torch.nn as nn
import pennylane as qml
from math import ceil

torch.manual_seed(0)

# 量子比特数和量子层数
n_qubits = 4  # 设置为4个量子比特
n_layers = 1

# 定义量子设备
dev = qml.device("default.qubit", wires=n_qubits)


# 量子电路
def quantum_circuit(inputs, weights):
    # 确保输入的大小符合量子比特数
    inputs = inputs[:n_qubits * (len(inputs) // n_qubits)]  # 截断输入以适应量子比特数
    var_per_qubit = len(inputs) // n_qubits
    encoding_gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)

    for qubit in range(n_qubits):
        qml.Hadamard(wires=qubit)
        for i in range(var_per_qubit):
            if (qubit * var_per_qubit + i) < len(inputs):
                if encoding_gates[i] == 'RZ':
                    qml.RZ(inputs[qubit * var_per_qubit + i], wires=qubit)
                elif encoding_gates[i] == 'RY':
                    qml.RY(inputs[qubit * var_per_qubit + i], wires=qubit)

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires=j % n_qubits)

    # 期望值测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# 量子神经网络模块
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        qnode = qml.QNode(quantum_circuit, dev, interface='torch', diff_method='best')
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        print("Input shape before flatten:", x.shape)

        # 展平输入并传递给量子层
        x = torch.flatten(x, start_dim=1)
        print("Input shape after flatten:", x.shape)

        # 确保输入大小适配量子层的期望
        required_size = (x.shape[1] // n_qubits) * n_qubits
        if x.shape[1] > required_size:
            x = x[:, :required_size]  # 截断多余的部分
        elif x.shape[1] < required_size:
            padding = required_size - x.shape[1]
            x = torch.cat([x, torch.zeros(x.shape[0], padding)], dim=1)  # 填充0到适应的大小

        print("Input shape before passing to quantum layer:", x.shape)

        # 在此之后不要再进行形状调整操作，确保量子层处理正确的输入大小
        return self.qlayer(x)
