import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
	def __init__(self, d_head, max_seq_len=2048, base=10000):
		super().__init__()
		self.d_head = d_head
		self.max_seq_len = max_seq_len
		self.base = base
		inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
		self.register_buffer("inv_freq", inv_freq)

		# Precompute the sinusoidal embeddings
		self._build_cache(max_seq_len)

	def _build_cache(self, max_seq_len):
		self.max_seq_len_cached = max_seq_len
		t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
		freqs = torch.einsum("i,j->ij", t, self.inv_freq)
		# Different from paper, but it uses a different permutation in order to obtain the same calculation
		emb = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
		self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

	def forward(self, q, k):
		# q, k: [bs, num_heads, seq_len, d_head]
		seq_len = q.shape[-2]
		if seq_len > self.max_seq_len_cached:
			self._build_cache(seq_len)

		# Get the cached cos and sin
		cos = self.cos_cached[:, :, :seq_len, ...]
		sin = self.sin_cached[:, :, :seq_len, ...]

		# Apply rotary embeddings
		q_out = (q * cos) + (self._rotate_half(q) * sin)
		k_out = (k * cos) + (self._rotate_half(k) * sin)
		return q_out, k_out

	def _rotate_half(self, x):
		# Rotates half the hidden dims of the input
		x1 = x[..., : self.d_head // 2]
		x2 = x[..., self.d_head // 2 :]
		return torch.cat((-x2, x1), dim=-1)
