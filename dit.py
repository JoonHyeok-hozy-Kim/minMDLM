import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size)
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device) # (dim//2)
    args = t[:, None] * freqs[None] # (batch_size x 1) * (1 x (dim//2)) = batch_size x (dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # batch_size x dim
    if dim % 2:
      embedding = torch.cat(
          [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
      )
    return embedding  # (batch_size x dim)

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
        dtype=next(self.parameters()).dtype
    )
    t_emb = self.mlp(t_freq)  # (batch_size x hidden_size)
    return t_emb


class Attention(nn.Module):
  def __init__(self, dim, n_heads):
    super().__init__()

    self.n_heads = n_heads
    self.n_rep = 1
    self.head_dim = dim // n_heads

    self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    # For RoPE in Llama 2
    self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
    self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

  # For RoPE in Llama 2
  @staticmethod
  def reshape_for_broadcast(freqs_cis, x):
      ndim = x.ndim
      assert 0 <= 1 < ndim
      # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
      _freqs_cis = freqs_cis[: x.shape[1]]
      shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
      return _freqs_cis.view(*shape)

  # For RoPE in Llama 2
  @staticmethod
  def apply_rotary_emb(xq, xk, freqs_cis):
      xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
      xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
      freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
      freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

      xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
      xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
      return xq_out, xk_out

  def forward(self, x, freqs_cis, attention_mask=None):
    batch_size, seq_len, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    dtype = xq.dtype

    # Layer Normalization
    xq, xk = self.q_norm(xq), self.k_norm(xk)

    # Reshape dim into n_heads * head_dim
    xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
    xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
    xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

    # Apply RoPE
    xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    xq, xk = xq.to(dtype), xk.to(dtype)

    # arg seq : batch_size, n_heads, seq_len, head_dim
    # xq * xv : batch_size, n_heads, seq_len^2
    # softmax * xv : batch_size, n_heads, seq_len, head_dim

    # print(f"xq shape : {xq.shape} vs attention_mask shape : {attention_mask.shape}")
    output = F.scaled_dot_product_attention(
        xq.permute(0, 2, 1, 3),
        xk.permute(0, 2, 1, 3),
        xv.permute(0, 2, 1, 3),
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,  # Setting for DDM (i.e. Non AR!)
    ).permute(0, 2, 1, 3) # Recovered to batch_size, seq_len, n_heads, head_dim
    output = output.flatten(-2) # Recovered to batch_size, seq_len, dim

    return self.wo(output)


# Llama SwiGLU FFN
class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Upsample and SiLU
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Downsample
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Upsample

  def _forward_silu_gating(self, x1, x3):
    return F.silu(x1) * x3

  def forward(self, x):
    return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
  def __init__(
      self,
      layer_id,
      dim,
      n_heads,
      multiple_of,
      ffn_dim_multiplier,
      norm_eps,
  ):
    super().__init__()
    self.dim = dim
    self.head_dim = dim // n_heads
    self.attention = Attention(self.dim, n_heads)
    self.feed_forward = FeedForward(
        dim=self.dim,
        hidden_dim=self.dim * 4,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = nn.LayerNorm(self.dim, eps=norm_eps)
    self.ffn_norm = nn.LayerNorm(self.dim, eps=norm_eps)

    # adaLN-Zero from DiT
    self.adaLN_modulation = nn.Sequential(
        nn.SiLU(),
        nn.Linear(min(dim, 1024), 6 * dim, bias=True),  # [shift, scale, gate] for MSA and FFN
    )

  def forward(self, x, freq_cis, adaln_input=None, attention_mask=None):
    if adaln_input is not None:
      # [shift, scale, gate] for Multi-head Self-attention(MSA) and FFN
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
          self.adaLN_modulation(adaln_input).chunk(6, dim=-1)
      )
      # MSA
      x = x + gate_msa.unsqueeze(1) * self.attention(
          modulate(self.attention_norm(x), shift_msa, scale_msa),
          freq_cis,
          attention_mask,
      )
      # FFN
      x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
          modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
      )
    else:
      x = x + self.attention(self.attention_norm(x), freq_cis, attention_mask)
      x = x + self.feed_forward(self.ffn_norm(x))

    return x

# Final Layer for the text token model
class FinalLayer(nn.Module):
  def __init__(self, hidden_size, vocab_size):
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) # no learning!
    # Output size : # of vocabulary
    self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
    # adaLN from DiT
    self.adaLN_modulation = nn.Sequential(
        nn.SiLU(),
        nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),  # [shift, scale]
    )

    # Zero-Intialization on weights (Check!)
    nn.init.constant_(self.linear.weight, 0.0)
    nn.init.constant_(self.linear.bias, 0.0)

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    x = modulate(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DiT_Llama(nn.Module):
  def __init__(
      self,
      vocab_size=32000,
      max_seq_len=1024,
      dim=512,
      n_layers=5,
      n_heads=16,
      multiple_of=256,
      ffn_dim_multiplier=None,
      norm_eps=1e-5,
  ):
    super().__init__()

    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len

    # Sequence
    self.x_embedder = nn.Embedding(vocab_size, dim)
    # nn.init.constant_(self.x_embedder.bias, 0)

    # Time Step
    self.t_embedder = TimestepEmbedding(min(dim, 1024))

    self.layers = nn.ModuleList(
        [
            TransformerBlock(
                layer_id,
                dim,
                n_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
            ) for layer_id in range(n_layers)
        ]
    )
    self.final_layer = FinalLayer(dim, vocab_size)
    self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, max_seq_len)  # (max_seq_len, dim // n_heads)

  def forward(self, x, t, attention_mask=None):
    x = self.x_embedder(x)  # (batch_size, seq_len, dim)
    batch_size, seq_len, _ = x.shape
    # Slice with the data sequence length
    freqs_cis_slice = self.freqs_cis[:seq_len].to(x.device) # (seq_len, dim // n_heads)

    t = self.t_embedder(t)  # (batch_size, min(dim,1024))
    adaln_input = t.to(x.dtype)

    # Attention mask preprocess (Match size and fill logit values)
    model_mask = None
    if attention_mask is not None:
      model_mask = torch.zeros_like(attention_mask, dtype=x.dtype)
      model_mask = model_mask.masked_fill(attention_mask == 0.0, -torch.inf)
      model_mask = model_mask.unsqueeze(1).unsqueeze(2) # (BATCH_SIZE,1,1,MAX_SEQ_LEN)

    for layer in self.layers:
      x = layer(x, freqs_cis_slice, adaln_input, model_mask)

    x = self.final_layer(x, adaln_input)
    return x

  @staticmethod
  def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)) # (dim//2)
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float() # (end, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # (end, dim//2)
    return freqs_cis