import torch
import torch.nn.functional as F
from torch import nn
import policy_attention_map as pam


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, model_dim: int, out_dim: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.p_enc = nn.Linear(input_dim, model_dim)
        self.layerNorm1 = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, num_heads * model_dim)
        self.layerNorm2 = nn.LayerNorm(model_dim)
        self.p_dec = nn.Linear(model_dim, out_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.size()
        x = self.p_enc(x)
        x = self.layerNorm1(x)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, self.num_heads, self.model_dim).permute(0, 2, 1, 3)
        x = self.layerNorm2(x)
        x = self.p_dec(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, model_dim: int, num_heads: int, p_enc_dim: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.model_dim == self.num_heads * self.head_dim, "model_dim must be num_heads * head_dim"

        self.qkv_linear = nn.Linear(input_dim, model_dim * 3)
        self.out_linear = nn.Linear(model_dim, input_dim)
        self.p_enc = PositionalEncoding(input_dim, num_heads, p_enc_dim, seq_len)

    def forward(self, x: torch.tensor) -> torch.tensor:
        qkv = self.qkv_linear(x)
        qkv = qkv.view(-1, self.seq_len, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        dk = torch.tensor(q.size(-1), dtype=torch.float32, device=q.device)
        attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(dk)
        p_enc = self.p_enc(x)
        attention = F.softmax(attention + p_enc, dim=-1)
        # # visualize attention head #1
        # attention_head0 = attention[0][0].detach().cpu().numpy()
        # files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        # for i in range(8, 16):
        #     print(files[i % 8], i // 8 + 1)
        #     plt.imshow(attention_head0[i].reshape(8, 8))
        #     plt.scatter([i % 8], [i // 8], c='red', s=100)
        #     plt.show()
        out = torch.matmul(attention, v)

        out = out.permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.head_dim * self.num_heads)
        return self.out_linear(out)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, model_dim: int, num_heads: int, p_enc_dim: int, dim_feedforward: int) -> None:
        super(MultiHeadAttentionModule, self).__init__()
        self.multi_head_attention = MultiHeadAttention(seq_len, input_dim, model_dim, num_heads, p_enc_dim)
        self.layerNorm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.layerNorm2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        x = self.multi_head_attention(x)
        x = x + residual
        x = self.layerNorm1(x)
        residual = x
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = x + residual
        x = self.layerNorm2(x)
        return x


class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, position_embedding_dim: int, out_embedding_dim: int, seq_len: int, attention_map: torch.tensor) -> None:
        super(InputEmbedding, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.register_buffer('attention_map', attention_map)
        self.embedding = nn.Linear(input_dim, position_embedding_dim)
        self.layerNorm1 = nn.LayerNorm(position_embedding_dim + input_dim)
        self.relu1 = nn.ReLU()
        map_dim = attention_map.shape[-1]

        self.linear1 = nn.Linear(map_dim + input_dim + position_embedding_dim, out_embedding_dim)
        self.layerNorm2 = nn.LayerNorm(out_embedding_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        if x.dim() > 3:
            dim1, dim2, dim3, dim4 = x.size()
            x = x.view(dim1, dim2, dim3 * dim4)
        batch_size, _, _ = x.size()
        x = x.view(batch_size, self.seq_len, self.input_dim)

        embedding = self.embedding(x)
        x = torch.cat((x, embedding), dim=-1)
        x = self.layerNorm1(x)
        x = self.relu1(x)
        map = self.attention_map.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device, dtype=x.dtype)
        x = torch.cat((x, map), dim=-1)
        x = self.linear1(x)
        x = self.layerNorm2(x)
        return self.relu2(x)


class PolicyHead(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, model_dim: int, num_heads: int, policy_dim: int, attention_map: torch.tensor) -> None:
        super(PolicyHead, self).__init__()
        self.register_buffer('attention_map', attention_map)
        map_dim = attention_map.shape[-1]
        assert seq_len == attention_map.shape[-2], "sequence length must be equal to map length"

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.model_dim == self.num_heads * self.head_dim, "model_dim must be num_heads * head_dim"

        self.kv_linear = nn.Linear(input_dim, model_dim * 2)
        self.q_linear = nn.Linear(map_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, input_dim)
        self.layerNorm = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()
        self.policy_linear = nn.Linear(input_dim, policy_dim)
        self.flatten_map = nn.Flatten()

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, _, _ = x.size()

        residual = x
        kv = self.kv_linear(x)
        map = self.attention_map.to(x.device, dtype=x.dtype)
        q = self.q_linear(map.repeat(batch_size, 1, 1))
        kv = kv.view(batch_size, self.seq_len, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        q = q.view(batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        dk = torch.tensor(q.size(-1), dtype=torch.float32, device=q.device)
        attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(dk)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.seq_len, self.head_dim * self.num_heads)
        out = self.out_linear(out)

        out = out + residual
        out = self.layerNorm(out)
        out = self.relu(out)

        out = self.policy_linear(out)
        return self.flatten_map(out)


class ValueHead(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, model_dim: int) -> None:
        super(ValueHead, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model_dim = model_dim

        self.linear1 = nn.Linear(input_dim, model_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(seq_len * model_dim, seq_len)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(seq_len, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = x.view(-1, self.seq_len * self.model_dim)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return self.tanh(x)


class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, position_embedding_dim: int, embedding_dim: int, seq_len: int, hidden_layers: int, attention_dim: int,
                 attention_heads: int,
                 positional_encoding_dim: int, dim_feedforward: int, value_model_dim: int, policy_model_dim: int, policy_out_dim: int) -> None:
        super(TransformerNet, self).__init__()
        attention_map = torch.tensor(pam.get_attention_map(), dtype=torch.float32)

        self.input_embedding = InputEmbedding(input_dim, position_embedding_dim, embedding_dim, seq_len, attention_map)
        self.attentionBlocks = nn.ModuleList(
            [MultiHeadAttentionModule(seq_len, embedding_dim, attention_dim, attention_heads, positional_encoding_dim,
                                      dim_feedforward) for _ in range(hidden_layers)]
        )
        self.value_head = ValueHead(seq_len, embedding_dim, value_model_dim)
        self.policy_head = PolicyHead(seq_len, embedding_dim, policy_model_dim, attention_heads, policy_out_dim,
                                      attention_map)

    def forward(self, x: torch.tensor) -> tuple:
        x = self.input_embedding(x)
        for attentionBlock in self.attentionBlocks:
            x = attentionBlock(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return policy.squeeze(-1), value.squeeze(-1)
