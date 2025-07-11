from utils.gumbel_softmax import gumbel_softmax
from layers.GCN import GCN
from layers.MCD import MCD
import pywt  # 需要安装 PyWavelets 库
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import sys

sys.path.append("..")


class D_R(nn.Module):
    def __init__(self, input_dim, input_len, total_level, current_level, dropout, enc_hidden, output_len, K_IMP):
        super().__init__()
        self.current_level = current_level
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_size = enc_hidden
        self.output_len = output_len
        self.dropout = dropout
        self.K_IMP = K_IMP
        self.total_level = total_level
        # 预测层
        self.forecast_proj1 = nn.ModuleList([nn.Linear(input_len, enc_hidden) for _ in range(2 ** total_level * K_IMP)])
        self.forecast_proj2 = nn.ModuleList([nn.Linear(enc_hidden, input_len) for _ in range(2 ** total_level * K_IMP)])
        self.activate = nn.LeakyReLU()

        # 高频和低频的可训练权重
        self.low_freq_weight = nn.Parameter(torch.randn(input_len // 2 + 1))
        self.high_freq_weight = nn.Parameter(torch.randn(input_len // 2 + 1))

        self.MCD = MCD(K_IMP, kernel_size=(1, 3), soft_max=False)

        if current_level == 0:
            pass
        else:

            self.branch_slelect = nn.Sequential(
                nn.Linear(input_len, 64),
                nn.BatchNorm2d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Linear(64, 2),
            )

            self.bsmask_conv = nn.Conv2d(
                in_channels=1,
                out_channels=K_IMP,
                kernel_size=(K_IMP, 3),
                stride=1,
                padding=(0, 1)
            )

            self.reconstruct_proj_left = nn.Sequential(
                nn.Linear(input_len, self.hidden_size),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, input_len),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU()
            )

            self.reconstruct_proj_right = nn.Sequential(
                nn.Linear(input_len, self.hidden_size),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, input_len),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU()
            )

            self.D_R_left = D_R(
                input_dim, input_len, total_level, current_level - 1, dropout, enc_hidden, output_len, K_IMP)
            self.D_R_right = D_R(
                input_dim, input_len, total_level, current_level - 1, dropout, enc_hidden, output_len, K_IMP)

            if current_level == total_level:
                self.forecast_proj1 = nn.ModuleList(
                    [nn.Linear(self.input_len, self.hidden_size)
                     for i in range(2 ** total_level * self.K_IMP)]
                )
                self.activate = nn.LeakyReLU()
                self.forecast_proj2 = nn.ModuleList(
                    [nn.Linear(self.hidden_size, self.input_len)
                     for i in range(2 ** total_level * self.K_IMP)]
                )
            else:
                pass

    def decompose_MCD(self, x):
        x = self.MCD(x)
        return x  # (B, N, K, T)

    def reconstruct(self, x_imf):  # (B, N, K, T)

        select_feature = self.branch_slelect(x_imf)  # (B, N, K, 2)

        B, N, K, T = x_imf.shape
        x_imf = x_imf.reshape(B * N, 1, K, T)  # (B, N, K, T) -> (B*N, 1, K, T)
        # (B*N, 1, K, T) conv-> (B*N, K, 1, T) permute-> (B*N, 1, K, T)
        imf_mask = self.bsmask_conv(x_imf).permute(0, 2, 1, 3)  # 指导掩码，对时间维度自适应调整，
        '''
        这个引导掩码的作用是用于在时间维度上进行自适应调整。它通过引导掩码对不同频率成分进行加权，以便对分量进行动态分离和聚合。这种机制关注的是时间序列中局部性和时间变化特性，旨在根据时间上的特定模式将数据分配到合适的分支上。
        该机制的目的在于适应输入序列的非平稳特性。通过选择合适的分支，引导掩码能够帮助模型更灵活地处理不同的时间变化模式，从而更好地建模时间序列的短期和长期依赖关系。
        '''

        '''
        傅里叶变换的作用是将时间序列的信号分解为不同的频率分量。通过将频率分量划分为低频和高频，模型能够捕捉数据中的周期性特征，并使用可训练的权重对不同频率成分进行加权。这样，傅里叶变换能增强模型对周期性特征的捕捉，使其在处理含有周期性模式的数据时更加有效。

        总的来说傅里叶变换关注的是全局的频率成分，与引导掩码主要针对时间维度上的局部调整不同
        '''
        x_imf = x_imf * imf_mask
        x_imf = x_imf.reshape(B, N, K, T)

        # (B, N, K, 2) one_hot if hard = True
        hard_class = gumbel_softmax(select_feature, hard=False)
        x_imf = x_imf.permute(0, 1, 3, 2)  # (B, N, T, K)
        x_summed = torch.matmul(x_imf, hard_class)  # (B, N, T, 2)

        x_left = self.reconstruct_proj_left(
            x_summed[:, :, :, 0])  # (B, N, T) --> # (B, N, T)
        x_right = self.reconstruct_proj_right(x_summed[:, :, :, 1])

        return x_left, x_right

    '''
           MCD对频率的分解，已经把数据分解为低频（趋势）和高频（季节性）部分。这样一来，MCD 本身就已经在一定程度上捕捉了数据的趋势性和季节性特征。
           '''

    def fourier_time_feature(self, x):  # 现在就是，这段代码会有些多余，因为也是对频率上的改动
        B, N, K, T = x.shape

        # 对最后一个时间维度进行傅里叶变换
        fft_result = torch.fft.rfft(x, dim=-1)

        # 计算幅值和相位
        amplitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        # 将频率划分为低频和高频成分
        half_idx = amplitude.shape[-1] // 2
        low_freq = amplitude[..., :half_idx]
        high_freq = amplitude[..., half_idx:]

        # 应用可训练权重调整低频和高频的比例
        weighted_low_freq = low_freq * self.low_freq_weight[:half_idx].view(1, 1, 1, -1)
        weighted_high_freq = high_freq * self.high_freq_weight[:amplitude.shape[-1] - half_idx].view(1, 1, 1, -1)

        # 合并高频和低频，并重建傅里叶特征
        weighted_amplitude = torch.cat([weighted_low_freq, weighted_high_freq], dim=-1)
        weighted_fft_result = weighted_amplitude * torch.exp(1j * phase)
        fourier_feature = torch.fft.irfft(weighted_fft_result, n=T, dim=-1)

        return fourier_feature

    def forecast(self, total_imf):
        B, N, K, T = total_imf.shape

        # 使用傅里叶变换提取增强的周期性特征
        fourier_feature = self.fourier_time_feature(total_imf)

        # 残差连接，叠加原始特征和傅里叶增强特征
        total_imf = total_imf + fourier_feature

        # 原始预测代码
        y_imf = None
        for i in range(2 ** self.total_level * self.K_IMP):
            y_current_imf = self.forecast_proj2[i](
                self.activate(
                    self.forecast_proj1[i](total_imf[:, :, i, :])
                )
            )
            y_current_imf = y_current_imf.unsqueeze(2)
            y_imf = y_current_imf if y_imf is None else torch.cat((y_imf, y_current_imf), axis=2)

        return y_imf  # (B, N, 1, T)

    def forward(self, x):
        if self.current_level == 0:
            x_imf = self.decompose_MCD(x)
            return x_imf  # (B,N,K,T)
        else:
            x_imf = self.decompose_MCD(x)
            x_left, x_right = self.reconstruct(x_imf)
            imf_left = self.D_R_left(x_left)
            imf_right = self.D_R_right(x_right)
            # 2*(B,N,2^(level-1)*K,T) --> (B,N,2^level*K,T)
            total_imf = torch.cat([imf_left, imf_right], dim=2)

            if self.current_level == self.total_level:
                y = self.forecast(total_imf)
                return y
            else:
                return total_imf


class IFNet(nn.Module):
    def __init__(self, output_len, input_len, dec_hidden=1024, dropout=0.5):
        super(IFNet, self).__init__()
        self.gcn = GCN(in_features=input_len,
                                    hidden_features=dec_hidden,
                                    out_features=input_len,
                                    latent_dim=32,
                                    num_heads=4)  # 这里添加 num_heads 参数

        self.activate = nn.LeakyReLU()
        self.predict = nn.Linear(input_len, output_len)

    def forward(self, x):
        # skip_x = self.gcn(x)
        # x = skip_x + x
        x = self.gcn(x)
        x = self.activate(x)
        x = torch.sum(x, 2)
        x = self.predict(x)
        return x


class DPAD(nn.Module):
    def __init__(self, input_dim, input_len, num_levels, dropout, enc_hidden, dec_hidden, output_len, K_IMP):
        super().__init__()
        self.levels = num_levels
        self.D_R_D = D_R(
            input_dim=input_dim,
            input_len=input_len,
            total_level=num_levels - 1,
            current_level=num_levels - 1,
            dropout=dropout,
            enc_hidden=enc_hidden,
            output_len=output_len,
            K_IMP=K_IMP
        )
        self.IF = IFNet(
            input_len=input_len,
            output_len=output_len,
            dec_hidden=dec_hidden
        )

    def forward(self, x):
        x = self.D_R_D(x)  # (B, N, 2^level-1, T)

        x = self.IF(x)

        return x


class DPAD_GCN(nn.Module):
    def __init__(self, output_len, input_len, input_dim=9, enc_hidden=1, dec_hidden=1, num_levels=3, dropout=0.5,
                 K_IMP=6, RIN=0):
        super(DPAD_GCN, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.num_levels = num_levels
        self.dropout = dropout
        self.K_IMP = K_IMP
        self.RIN = RIN

        self.DPAD = DPAD(
            input_dim=self.input_dim,
            input_len=self.input_len,
            num_levels=self.num_levels,
            dropout=self.dropout,
            enc_hidden=self.enc_hidden,
            dec_hidden=self.dec_hidden,
            output_len=output_len,
            K_IMP=self.K_IMP
        )

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x):

        ### activated when RIN flag is set ###
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias

        x = x.permute(0, 2, 1)  # (B,N,T)
        x = self.DPAD(x)
        x = x.permute(0, 2, 1)  # (B,T,N)

        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden-size', default=1,
                        type=int, help='hidden size of module')
    parser.add_argument('--RIN', default=1, type=int, help='ReVIN')
    args = parser.parse_args()
    model = DPAD_GCN(input_len=168, output_len=args.horizon, input_dim=8, enc_hidden=168,
                     dec_hidden=168, dropout=0.5, num_levels=2, K_IMP=6, RIN=1).cuda()

    x = torch.randn(32, 168, 8).cuda()

    y = model(x)

    print(y.shape)
