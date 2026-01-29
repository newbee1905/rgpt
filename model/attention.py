import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from mdoel.rope import RotaryEmbedding
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

		self.qk_norm = QKNorm(self.d_head, config.norm_type, config.norm_eps) if config.qk_norm else None

		if self.g1_gate:
			self.gate_proj = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=self.n_heads, bias=False)

		self.rotary_emb = RotaryEmbedding(self.d_head, max_seq_len=config.max_seq_len)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		batch_size, seq_len, _ = x.shape

		q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
		k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
		v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

		if self.qk_norm:
			q, k = self.qk_norm(q, k)

		q, k = self.rotary_emb(q, k)

		attn_output = F.scaled_dot_product_attention(
			q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
		)

		if self.g1_gate:
			q_for_gate = q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model).transpose(1, 2)
			gate_scores = self.gate_proj(q_for_gate)
			gate_scores = (
				gate_scores.transpose(1, 2).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
			)
			attn_output = attn_output * torch.sigmoid(gate_scores)

		attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
		return self.out_proj(attn_output)
