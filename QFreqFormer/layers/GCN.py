import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, latent_dim, num_heads):
        super(GCN, self).__init__()
        # 节点特征的潜在空间嵌入
        self.latent_embed = nn.Linear(in_features, latent_dim)

        # 多头注意力机制
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(num_heads)]
        )

        # 原始特征的嵌入层
        self.feature_embed = nn.Linear(in_features, hidden_features)

        # 输出层
        self.output_layer = nn.Linear(hidden_features, out_features)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, N, K, T), B=batch size, N=nodes, K=features, T=sequence length

        # 将输入数据映射到潜在空间
        latent_x = self.latent_embed(x)  # (B, N, K, latent_dim)

        # 计算多头注意力
        attention_matrices = []
        for i in range(self.num_heads):
            head_x = self.attention_heads[i](latent_x)  # (B, N, K, latent_dim)
            head_x_t = head_x.permute(0, 1, 3, 2)  # (B, N, latent_dim, K)
            attention_scores = torch.matmul(head_x, head_x_t)  # (B, N, K, K)
            attention_scores = F.softmax(attention_scores, dim=-1)  # 归一化
            attention_matrices.append(attention_scores)

        # 合并多头注意力矩阵
        adj = torch.mean(torch.stack(attention_matrices), dim=0)  # (B, N, K, K)

        # 对原始特征进行线性变换
        x = self.feature_embed(x)  # (B, N, K, hidden_features)

        # 在潜在空间中的相似性矩阵上进行特征传播
        x = torch.matmul(adj, x)  # (B, N, K, hidden_features)

        # 输出层
        x = self.output_layer(x)  # (B, N, K, out_features)

        return x


if __name__ == '__main__':
    features = torch.randn(32, 8, 6, 96)
    gcn = GCN(in_features=96, hidden_features=16, out_features=2, latent_dim=32, num_heads=4)
    output = gcn(features)

    print(output.shape)
