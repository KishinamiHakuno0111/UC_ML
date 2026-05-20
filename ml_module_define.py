# DNN模型定义
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 定义maxout
class Maxout(nn.Module):
    def __init__(self, num_units, num_pieces):
        super(Maxout, self).__init__()
        self.num_units = num_units
        self.num_pieces = num_pieces

    def forward(self, x):
        # 将输入切分成 num_pieces 个部分
        maxout_input = torch.chunk(x, self.num_pieces, dim=1)
        # 对每个部分取最大值
        maxout_output = torch.stack(maxout_input, dim=-1).max(dim=-1)[0]
        return maxout_output

# 定义深度神经网络模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob, num_pieces, is_maxout):
        super(DNN, self).__init__()

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            if is_maxout:
                layers.append(nn.Linear(prev_size, size * num_pieces))
                layers.append(Maxout(size, num_pieces))  # 使用 Maxout 激活函数
            else:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())  # 用ReLU激活函数
            layers.append(nn.Dropout(p=dropout_prob)) #  加入dropout
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class KAN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob, num_pieces, is_maxout):
        """
        Kolmogorov-Arnold Network
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            num_hidden_layers (int): 隐藏层数量
        """
        super(KAN, self).__init__()

        # 定义输入层
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # 定义隐藏层
        hidden_layer_list = []
        for i in range(len(hidden_sizes)-1):
            hidden_layer_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            hidden_layer_list.append(nn.Dropout(p=dropout_prob))

        # hidden_layer_list.append(nn.Dropout(p=dropout_prob))
        self.hidden_layers = nn.ModuleList(hidden_layer_list)

        # 定义输出层
        self.output_layer = nn.Sequential(*[nn.Linear(hidden_sizes[i+1], output_size), nn.Sigmoid()])

    def forward(self, x):
        # 输入层 + 激活函数
        x = torch.sin(self.input_layer(x))

        # 经过每个隐藏层并加上非线性激活
        for layer in self.hidden_layers:
            x = torch.sin(layer(x))

        # 输出层
        output = self.output_layer(x)

        return output

# 定义双向LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(BiLSTM, self).__init__()

        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Dropout层
        self.dropout = nn.Dropout(p=dropout_prob)

        # 输出层，调整LSTM的输出维度为所需的输出维度
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向LSTM，所以hidden_size * 2

        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM层输出 (batch_size, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)

        # Dropout层
        lstm_out = self.dropout(lstm_out)

        # 通过全连接层调整输出维度
        output = self.fc(lstm_out)

        # 添加 Sigmoid 激活函数，使输出限制在 [0, 1] 之间
        output = self.sigmoid(output)

        return output

# class BiLSTM(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(BiLSTM, self).__init__()
#         self.hidden_sizes = hidden_sizes
#         self.num_layers = len(hidden_sizes)
#
#         # 创建多层双向LSTM
#         self.lstms = nn.ModuleList([nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
#                                     for hidden_size in hidden_sizes])
#
#         # 全连接层
#         self.fc = nn.Linear(sum(hidden_sizes) * 2, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 前向传播，逐层处理
#         lstm_outputs = []
#         for i in range(self.num_layers):
#             out, _ = self.lstms[i](x)
#             lstm_outputs.append(out)
#
#         # 拼接所有层的输出
#         n_dim = out.ndim
#         concatenated_outputs = torch.cat(lstm_outputs, dim=(n_dim-1))
#
#         # 全连接层
#         out = self.fc(concatenated_outputs[-1])
#         out = self.sigmoid(out)
#         return out

class BiLSTM_new(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BiLSTM_new, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # 创建多层双向LSTM
        self.lstms = nn.ModuleList([nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
                                    for hidden_size in hidden_sizes])

        # 全连接层
        self.fc = nn.Linear(hidden_sizes[0] * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 前向传播，逐层处理
        lstm_outputs = []
        for i in range(self.num_layers):
            out, _ = self.lstms[i](x[:, i, :])
            out = out.unsqueeze(1)
            lstm_outputs.append(out)

        # 拼接所有层的输出
        n_dim = out.ndim
        concatenated_outputs = torch.cat(lstm_outputs, dim=(n_dim-2))

        # 全连接层
        out = self.fc(concatenated_outputs)  # 原来是仍然取最后一个时间步的输出，现在取全时间段输出
        out = self.sigmoid(out)
        return out

# 定义transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        positional_info = self.encoding[:, :x.size(1), :].detach()
        positional_info = positional_info.to(x.device)
        return x + positional_info #  这里修改了，原本参见第5个模型


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        x = self.fc(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attended = self.attention(x, x, x, mask)
        a = self.dropout(attended)
        x = self.norm1(x + a) #  这里修改了,原本是直接相加
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size=89, output_size=12, seq_len=36, num_heads=8, hidden_dim=256, num_layers=6,
                 dropout=0.1):
        super(TransformerModel, self).__init__()

        # Embedding layers
        self.embedding_input = nn.Linear(input_size, hidden_dim)
        self.embedding_output = nn.Linear(hidden_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding_input(x)
        x = self.positional_encoding(x)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=None)  # Assuming no masking for simplicity

        # Output layer
        x = self.output_layer(x)

        # Apply activation function (e.g., sigmoid for binary classification)
        x = torch.sigmoid(x)

        return x
