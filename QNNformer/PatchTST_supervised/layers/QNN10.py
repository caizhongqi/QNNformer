import pennylane as qml
import torch
from torch import nn
import numpy as np

# 设置量子比特数量和层数
n_qubits = 4  # 量子比特的数量
n_layers = 3  # 量子层的深度

# 定义量子设备
dev = qml.device('default.qubit', wires=n_qubits)


def simplified_entanglement(wires, entanglement_weights):
    """简化的纠缠结构，去除长距离交互和复杂门操作"""
    n = len(wires)
    for i in range(n):
        qml.CRX(entanglement_weights[i], wires=[wires[i], wires[(i + 1) % n]])


def quantum_circuit_with_simplified_entanglement(inputs, weights, entanglement_weights):
    """简化后的量子电路，去除了复杂的隧穿和量子相位估计"""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    for layer in range(n_layers):
        # 量子旋转操作
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)

        # 动态纠缠
        simplified_entanglement(range(n_qubits), entanglement_weights[layer])

    # 测量PauliZ、PauliX和PauliY的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumAttentionWithSimplifiedEntanglement(nn.Module):
    def __init__(self, input_features, num_heads):
        super(QuantumAttentionWithSimplifiedEntanglement, self).__init__()
        self.num_heads = num_heads
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features})不能整除split_size ({self.split_size}).")

        # 为 Q、K、V 分别创建量子节点
        self.qnodes_q = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_simplified_entanglement, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        self.qnodes_k = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_simplified_entanglement, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

        self.qnodes_v = nn.ModuleList([
            qml.qnn.TorchLayer(
                qml.QNode(quantum_circuit_with_simplified_entanglement, dev, interface='torch', diff_method='best'),
                {"weights": (n_layers, self.split_size), "entanglement_weights": (n_layers, self.split_size)}
            ) for _ in range(self.n_splits)
        ])

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

        return attn_output, attn_weights


class QuantumLayer(nn.Module):
    def __init__(self, input_features, num_heads=4):
        super(QuantumLayer, self).__init__()
        self.split_size = n_qubits
        self.n_splits = input_features // self.split_size

        if input_features % self.split_size != 0:
            raise ValueError(f"Input features ({input_features})不能整除split_size ({self.split_size}).")

        # 创建简化的量子多头注意力模块
        self.q_attention = QuantumAttentionWithSimplifiedEntanglement(input_features, num_heads)

        # 初始化量子节点的权重使用Xavier初始化
        for qnode in self.q_attention.qnodes_q:
            nn.init.xavier_uniform_(qnode.weights)

        # 全连接层，减少参数量并加入L2正则化
        self.fc = nn.Sequential(
            nn.Linear(3 * input_features, input_features),  # 保持较少参数
            nn.ReLU(),
            nn.Dropout(p=0.1)  # 加入dropout防止过拟合
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        assert features == self.split_size * self.n_splits, f"Features size {features}不匹配split_size."

        # 通过简化的量子多头注意力机制处理输入
        attn_output, _ = self.q_attention(x)

        # 确保 attn_output 的形状与全连接层输入维度一致
        attn_output = attn_output.reshape(batch_size * seq_len, -1)

        # 计算全连接层期望的输入维度
        expected_input_features = self.fc[0].in_features  # 获取全连接层的输入特征数
        attn_output = attn_output[:, :expected_input_features]  # 保证 attn_output 的特征数符合全连接层的输入要求

        # 通过全连接层融合量子特征
        x = self.fc(attn_output)

        # 恢复形状为 [batch_size, seq_len, -1]
        x = x.reshape(batch_size, seq_len, -1)
        return x

