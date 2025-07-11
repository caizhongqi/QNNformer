import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalGraphAttentionLayer(nn.Module):
    """
    A Spatio-Temporal Graph Attention Layer (ST-GAT) for handling both spatial and temporal dependencies.
    """
    def __init__(self, input_dim, output_dim, num_nodes, num_timesteps, dropout, alpha):
        super(SpatioTemporalGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.alpha = alpha

        # Spatial Attention Parameters
        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Temporal Convolution
        self.temporal_conv = nn.Conv1d(num_timesteps, num_timesteps, kernel_size=3, padding=1)

        # Activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        x: Input features of shape (batch_size, num_nodes, num_timesteps, input_dim)
        adj: Adjacency matrix for spatial relations, shape (num_nodes, num_nodes)
        """
        batch_size, num_nodes, num_timesteps, input_dim = x.size()

        # Temporal Convolution
        x = x.permute(0, 2, 1, 3)  # Change to (batch_size, num_timesteps, num_nodes, input_dim)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1, 3)  # Back to (batch_size, num_nodes, num_timesteps, output_dim)

        # Reshape for Spatial Attention
        x = x.view(batch_size, num_nodes, -1)  # Flatten the temporal dimension
        Wh = torch.matmul(x, self.W)  # Spatial transformation (batch_size, num_nodes, output_dim)

        # Attention Mechanism
        a_input = self._prepare_attention_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # Masking with Adjacency Matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregation
        h_prime = torch.matmul(attention, Wh)  # (batch_size, num_nodes, output_dim)

        return F.elu(h_prime).view(batch_size, num_nodes, num_timesteps, self.output_dim)

    def _prepare_attention_input(self, Wh):
        """
        Prepare the input for attention mechanism.
        """
        B, N, OF = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # (B, N*N, OF)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # (B, N*N, OF)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, N, N, 2 * OF)


class STGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_timesteps, dropout=0.5, alpha=0.2):
        super(STGAT, self).__init__()
        self.spatial_temporal_layer1 = SpatioTemporalGraphAttentionLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_nodes=num_nodes,
            num_timesteps=num_timesteps,
            dropout=dropout,
            alpha=alpha
        )
        self.spatial_temporal_layer2 = SpatioTemporalGraphAttentionLayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            num_timesteps=num_timesteps,
            dropout=dropout,
            alpha=alpha
        )

    def forward(self, x, adj):
        x = self.spatial_temporal_layer1(x, adj)
        x = self.spatial_temporal_layer2(x, adj)
        return x


# Testing the STGAT model
if __name__ == '__main__':
    batch_size = 4
    num_nodes = 10
    num_timesteps = 5
    input_dim = 16
    hidden_dim = 8
    output_dim = 4

    x = torch.randn(batch_size, num_nodes, num_timesteps, input_dim)
    adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)  # Fully connected graph minus self-loops

    model = STGAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                  num_nodes=num_nodes, num_timesteps=num_timesteps, dropout=0.5, alpha=0.2)
    output = model(x, adj)
    print("Output shape:", output.shape)
