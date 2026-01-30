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

		self.max_seq_len_cached = 0
		self.cos_cached = None
		self.sin_cached = None

	def _build_cache(self, seq_len, device, dtype):
		if seq_len <= self.max_seq_len_cached and self.cos_cached is not None:
			return

		self.max_seq_len_cached = seq_len

		# Use device and dtype from inv_freq if not provided
		if device is None:
			device = self.inv_freq.device
		if dtype is None:
			dtype = self.inv_freq.dtype

		t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)

		freqs = torch.einsum("i,j->ij", t, self.inv_freq)
		emb = torch.cat((freqs, freqs), dim=-1)

		self.cos_cached = emb.cos()[None, None, :, :]
		self.sin_cached = emb.sin()[None, None, :, :]

	def forward(self, q, k):
		# q, k: [bs, num_heads, seq_len, d_head]
		seq_len = q.shape[-2]
		self._build_cache(seq_len, q.device, q.dtype)

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
