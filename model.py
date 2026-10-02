import os

import torch 
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

CUR_DIR = os.path.dirname(__file__)

@dataclass
class NSConfig:
    vocab_size: int = 32_000
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12  # attn hidden dim = 12 * 64 = 768
    head_dim: int = 64
    n_kv_heads: int = 2
    context_length: int = 2048
    hidden_dim: int = 2048

    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6

    use_cache: bool = True




# 1. Rope
def init_rope(cfg: NSConfig):

    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))

    pos = torch.arange(0, cfg.context_length, dtype=torch.float32)

    angles = torch.einsum("i,j -> ij", pos, inv_freq)

    angles = torch.cat([angles, angles], dim=-1)

    cos = angles.cos()
    sin = angles.sin()

    return cos, sin


def rotate_half(x):

    head_dim = x.shape[-1]
    assert head_dim % 2 == 0, "head dim must be even"
    x1, x2 = x[:, :, :, : head_dim // 2], x[:, :, :, head_dim // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin, position_offset=0):

    seq_len = q.shape[-2]
    cos = cos[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated.to(dtype=q.dtype), k_rotated.to(dtype=k.dtype)


# 2. GQA Attention
class GQAAttention(nn.Module):

    def __init__(self, cfg: NSConfig):
        super().__init__()

        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.cfg = cfg
        self.kv_group_size = cfg.n_heads // cfg.n_kv_heads

        self.w_q = nn.Linear(cfg.emb_dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.w_k = nn.Linear(cfg.emb_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.w_v = nn.Linear(cfg.emb_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.emb_dim, bias=False)


    def forward(self, x, cos, sin, past_key_value=None, use_cache=False, attention_mask=None):
        """
            x: [B, L, D]
            cos: [L, D]
            sin: [L, D]
            past_key_value: (k, v)  past_key_value是一个元组，元组的第一个元素是k，第二个元素是v
            v的形状是[B, L, H, D]，表示历史token的V值，K同理

            use_cache: bool

            attention_mask = (bsz, seq_len),  1=keep, 0=pad  seq_len = past_len + new_tokens
        """

        bsz, new_tokens, _ = x.shape

        q = self.w_q(x).view(bsz, new_tokens, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.w_k(x).view(bsz, new_tokens, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.w_v(x).view(bsz, new_tokens, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)

        if past_key_value is None:
            past_k = None
            past_v = None
            past_len = 0
        else:
            past_k, past_v = past_key_value
            past_len = past_k.shape[-2]

        q, k = apply_rope(q, k, cos, sin, position_offset=past_len)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v)


        k_attn = torch.repeat_interleave(k, self.kv_group_size, dim=1)
        v_attn = torch.repeat_interleave(v, self.kv_group_size, dim=1)

        total_len = k_attn.shape[-2]

        pad_mask = None
        # -------- 构造pad_mask
        if attention_mask is not None:
            assert attention_mask.shape[-1] == total_len, "attention_mask的长度必须和total_len长度一致" 

            pad_mask = torch.zeros_like(attention_mask, dtype=q.dtype)
            pad_mask = pad_mask.masked_fill(attention_mask == 0, -torch.inf)
            pad_mask = pad_mask[:, None, None, :]


        if past_len == 0:
            # 标准 causal attention
            attn_output = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=pad_mask,
                dropout_p=0.0,
                is_causal=True,
            )  # [B, H, L, D]

        elif new_tokens == 1:
            # Decode: Q=1 token, K/V = past_tokens + 1 token
            attn_output = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=pad_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:

            q_positions = torch.arange(past_len, past_len + new_tokens, device=x.device)
            k_positions = torch.arange(0, total_len, device=x.device)


            # new_tokens 表示未来的token
            future_mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)

            # q_postions = 
            #    Q5    5    5   5   5   5   5   5
            #    Q6    6    6   6   6   6   6   6
            #    Q7    7    7   7   7   7   7   7

            # k_postions = 
            #         K0   K1  K2  K3  K4  K5  K6  K7
            #          0    1   2   3   4   5   6   7
            #          0    1   2   3   4   5   6   7
            #          0    1   2   3   4   5   6   7
            #         ||
            #         ||
            #         \/
            #  future_mask =
            #          K0   K1  K2  K3  K4  K5  K6  K7
            #    Q5    0    0   0   0   0   0   1   1
            #    Q6    0    0   0   0   0   0   0   1
            #    Q7    0    0   0   0   0   0   0   0

            #  attn_mask =
            #          K0   K1  K2  K3  K4  K5  K6    K7
            #    Q5    0    0   0   0   0   0   inf   inf
            #    Q6    0    0   0   0   0   0   0     inf
            #    Q7    0    0   0   0   0   0   0     0

            # pad_mask =
            #          K0   K1  K2  K3  K4  K5  K6    K7
            #    Q5    0    0   0   0   0   0   inf   inf
            #    Q6    0    0   0   0   0   0   inf   inf
            #    Q7    0    0   0   0   0   0   inf   inf

            attn_mask = torch.zeros((new_tokens, total_len), dtype=q.dtype, device=x.device)

            attn_mask = attn_mask.masked_fill(future_mask, -torch.inf)

            attn_mask = attn_mask[None, None, :, :]
            
            if pad_mask is not None:
                attn_mask = attn_mask + pad_mask
            

            attn_output = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )


        context_v = attn_output.transpose(1, 2).contiguous().view(bsz, new_tokens, self.cfg.n_heads * self.cfg.head_dim)

        output = self.w_o(context_v)

        if use_cache:
            return output, present_key_value

        return output


# 3. RMS Norm
class RMSNorm(nn.Module):

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, x):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * self.weight


# 4. FFN
class FeedForward(nn.Module):

    def __init__(self, cfg: NSConfig):
        super().__init__()

        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = F.silu(x1) * x2

        return self.fc3(x)


# 5. Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self, cfg: NSConfig):

        super().__init__()

        self.cfg = cfg

        self.rms_norm1 = RMSNorm(cfg.emb_dim, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)

        self.rms_norm2 = RMSNorm(cfg.emb_dim, cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg)

    

    def forward(self, x, cos, sin, past_key_value=None, use_cache=False, attention_mask=None):

        residual = x
        x = self.rms_norm1(x)

        if use_cache:
            x, present_key_value = self.attn(x, cos, sin, past_key_value=past_key_value, use_cache=True, attention_mask=attention_mask)
        else:
            x = self.attn(x, cos, sin, past_key_value=None, use_cache=False, attention_mask=attention_mask)

        x = residual + x 

        residual = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = residual + x

        if use_cache:
            return x, present_key_value

        return x


# 6. NsModel
class NsModel(nn.Module):

    def __init__(self, cfg: NSConfig):

        super().__init__()

        self.cfg = cfg

        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        cos, sin = init_rope(cfg)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.layers = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        self.final_norm = RMSNorm(cfg.emb_dim, cfg.rms_norm_eps)
        self.lm_out = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, x, past_key_values=None, use_cache=False, attention_mask=None):
        """
        past_key_values/present_key_values 结构
        [
            "layer_0": (present_k, present_v),
            "layer_1": (present_k, present_v),
            ...
            "layer_n": (present_k, present_v),
        ]

        attention_mask: [B, L]  (1=keep, 0=pad)，训练时 L = 当前输入长度；
                    推理带 cache 时，L = past_len + new_tokens
        """

        bsz, seq_len = x.shape

        if past_key_values is None:
            past_len = 0
        else:
            past_len = past_key_values[0][0].shape[-2]

        if past_len + seq_len > self.cfg.context_length:
            raise ValueError(f"Input sequence length {seq_len} + past sequence length {past_len} exceeds max length {self.cfg.context_length}")


        x = self.emb(x)


        # 每层分别保存 KV cache
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * self.cfg.n_layers

            # >>> a = [(1, 2), (3,4)]
            # >>> b = a
            # >>> a = []  # NOTE: present_key_values每次decode时置为[]无影响，因为每次decode完成后，present_key_values已经赋值给past_key_values了
            # >>> b
            # [(1, 2), (3, 4)]

            present_key_values = []

        for layer_idx, layer in enumerate(self.layers):

            if use_cache:
                x, present_kv = layer(x, self.cos, self.sin, past_key_value=past_key_values[layer_idx], use_cache=True, attention_mask=attention_mask)
                present_key_values.append(present_kv)
            else:
                x = layer(x, self.cos, self.sin, attention_mask=attention_mask)

        x = self.final_norm(x)

        logits = self.lm_out(x)

        if use_cache:

            return logits, present_key_values

        return logits



if __name__ == "__main__":
    print(CUR_DIR)