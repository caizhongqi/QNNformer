import pennylane as qml
import torch
from torch import nn
import numpy as np

n_qubits = 4  # 量子比特数量
n_layers = 3  # 量子层深度

# 定义量子设备
dev = qml.device('default.qubit', wires=n_qubits)

def advanced_dynamic_entanglement(wires, entanglement_weights):
    """增强的动态纠缠结构，引入长距离交互和多重纠缠"""
    n = len(wires)

    # 引入长距离的交叉纠缠
    for i in range(n):
        qml.CRX(entanglement_weights[i], wires=[wires[i], wires[(i + 2) % n]])
        qml.CRX(entanglement_weights[i], wires=[wires[i], wires[(i + 3) % n]])

    # 引入复杂的 iSWAP 门和 Toffoli 门增强纠缠强度
    for i in range(n - 1):
        qml.ISWAP(wires=[wires[i], wires[i + 1]])
    qml.ISWAP(wires=[wires[-1], wires[0]])  # 闭环的长距离纠缠

    # 使用 Toffoli 门进行多重控制纠缠
    for i in range(0, n, 3):  # 每三个比特进行一次三重纠缠
        if i + 2 < n:
            qml.Toffoli(wires=[wires[i], wires[i + 1], wires[i + 2]])

def quantum_tunneling_effect(wires, tunneling_weights):
    """量子隧穿效应，通过 CRX 门模拟隧穿"""
    for i in range(len(wires) - 1):
        qml.CRX(tunneling_weights[i], wires=[wires[i], wires[i + 1]])
    qml.CRX(tunneling_weights[-1], wires=[wires[-1], wires[0]])

def quantum_phase_estimation(wires):
    """量子相位估计 (Quantum Phase Estimation)"""
    # 将输入态初始化为均匀叠加态
    for wire in wires:
        qml.Hadamard(wires=wire)

    # 施加受控相移门
    for i in range(len(wires)):
        for j in range(i):
            qml.ControlledPhaseShift(np.pi / 2 ** (i - j), wires=[wires[j], wires[i]])

    # 应用量子傅里叶逆变换
    for i in range(len(wires) // 2):
        qml.SWAP(wires=[wires[i], wires[len(wires) - 1 - i]])

    for i in range(len(wires)):
        qml.Hadamard(wires=i)
        for j in range(i):
            qml.CPhase(-np.pi / 2 ** (i - j), wires=[j, i])

def quantum_circuit_with_qpe_and_tunneling(inputs, weights, entanglement_weights, tunneling_weights):
    """改进后的量子电路，结合量子相位估计和量子隧穿"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        # 量子旋转操作
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # 动态纠缠
        advanced_dynamic_entanglement(range(n_qubits), entanglement_weights[layer])

        # 应用量子隧穿效应
        quantum_tunneling_effect(range(n_qubits), tunneling_weights[layer])

        # 应用量子相位估计算法
        quantum_phase_estimation(range(n_qubits))

    # 测量 PauliZ、PauliX 和 PauliY 的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

class QuantumAttentionWithQPEAndTunneling(nn.Module):
    def __init__(self, input_features, num_heads):
        super(QuantumAttentionWithQPEAndTunneling, self).__init__()
        self.num_heads = num_heads
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features})不能整除split_size ({self.split_size}).")

        # 为 Q、K、V 分别创建量子节点
        self.qnodes_q = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_qpe_and_tunneling, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size),
                 "tunneling_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        self.qnodes_k = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_qpe_and_tunneling, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size),
                 "tunneling_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        self.qnodes_v = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_qpe_and_tunneling, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size),
                 "tunneling_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        # 正则化层和归一化层
        self.dropout = nn.Dropout(p=0.1)  # Dropout 概率为0.1
        self.layer_norm = nn.LayerNorm(3 * self.split_size * self.n_splits)  # 正确设置 LayerNorm 的 shape

        # 经典的多头注意力机制
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"Features size {features}与split_size不匹配."

        q_outputs, k_outputs, v_outputs = [], [], []

        # 分别生成 Q, K, V
        for i in range(self.n_splits):
            x_split = x[:, :, i * self.split_size:(i + 1) * self.split_size]
            x_split = x_split.reshape(-1, self.split_size)

            q_output = self.qnodes_q[i](x_split)
            k_output = self.qnodes_k[i](x_split)
            v_output = self.qnodes_v[i](x_split)

            q_outputs.append(q_output)
            k_outputs.append(k_output)
            v_outputs.append(v_output)

        # 拼接 Q, K, V
        query = torch.cat(q_outputs, dim=-1).reshape(batch_size, seq_len, -1)
        key = torch.cat(k_outputs, dim=-1).reshape(batch_size, seq_len, -1)
        value = torch.cat(v_outputs, dim=-1).reshape(batch_size, seq_len, -1)

        # 计算 Q 和 K 的相似度
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.split_size)
        attn_weights = self.softmax(attn_scores)

        # 使用注意力权重对 V 进行加权求和
        attn_output = torch.matmul(attn_weights, value)

        # 加入 Dropout 和 Layer Normalization
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)

        return attn_output, attn_weights


class QuantumLayer(nn.Module):
    def __init__(self, input_features, num_heads=4):
        super(QuantumLayer, self).__init__()
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features})不能整除split_size ({self.split_size}).")

        # 创建结合 QPE 和量子隧穿的量子多头注意力模块
        self.q_attention = QuantumAttentionWithQPEAndTunneling(input_features, num_heads)

        # 初始化量子节点的权重
        for qnode in self.q_attention.qnodes_q:
            qnode.weights.data = torch.tensor(np.random.uniform(low=-0.01, high=0.01, size=qnode.weights.shape))

        # 全连接层，维持较少参数，融合量子特征
        self.fc = nn.Sequential(
            nn.Linear(3 * input_features, input_features),  # 使用较少参数
            nn.ReLU()
        )

        # 正则化层
        self.dropout = nn.Dropout(p=0.001)  # Dropout 概率为0.1
        self.layer_norm = nn.LayerNorm(input_features)  # Layer Normalization

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"Features size {features}不匹配split_size."

        # 通过结合 QPE 和量子隧穿的量子多头注意力机制处理输入
        attn_output, _ = self.q_attention(x)

        # 确保 attn_output 的形状与全连接层输入维度一致
        attn_output = attn_output.reshape(batch_size * seq_len, -1)

        # 通过全连接层融合量子特征
        x = self.fc(attn_output)

        # 加入 Dropout 和 Layer Normalization
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 恢复形状为 [batch_size, seq_len, -1]
        x = x.reshape(batch_size, seq_len, -1)
        return x
