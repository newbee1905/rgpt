import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from model.rope import RotaryEmbedding
from model.norm import QKNorm


class Attention(nn.Module):
	def __init__(self, config: DictConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.d_model = config.d_model
		assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
		self.d_head = self.d_model // self.n_heads
		self.g1_gate = config.g1_gate

		self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

		self.qk_norm = (
			QKNorm(self.d_head, config.qk_normtype, config.norm_eps) if config.qk_normtype != "none" else None
		)

		if self.g1_gate:
			self.gate_proj = nn.Linear(self.d_head, self.d_head, bias=True)
			nn.init.constant_(self.gate_proj.bias, 1.0)

		self.rotary_emb = RotaryEmbedding(self.d_head, max_seq_len=config.max_seq_len)
		self.dropout_p = config.dropout

	def forward(self, x):
		bsz, seq_len, _ = x.shape

		q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)
		k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)
		v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)

		if self.qk_norm:
			q, k = self.qk_norm(q, k)

		gate = None
		if self.g1_gate:
			gate = torch.sigmoid(self.gate_proj(q))
			gate = gate.transpose(1, 2)

		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)

		q, k = self.rotary_emb(q, k)

		attn_output = F.scaled_dot_product_attention(
			q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
		)

		if self.g1_gate:
			attn_output = attn_output * gate

		attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
		return self.out_proj(attn_output)

	def forward_with_attention(self, x):
		bsz, seq_len, _ = x.shape

		q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)
		k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)
		v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)

		if self.qk_norm:
			q, k = self.qk_norm(q, k)

		gate = None
		if self.g1_gate:
			gate = torch.sigmoid(self.gate_proj(q))
			gate = gate.transpose(1, 2)

		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)

		q, k = self.rotary_emb(q, k)

		# Manual attention to get weights
		attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
		mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
		attn_logits.masked_fill_(mask[None, None, :, :], float("-inf"))
		attn_weights = F.softmax(attn_logits, dim=-1)
		attn_output = torch.matmul(attn_weights, v)

		if self.g1_gate:
			attn_output = attn_output * gate

		attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
		attn_output = self.out_proj(attn_output)

		return attn_output, attn_weights
