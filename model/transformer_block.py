import torch
import torch.nn as nn
from omegaconf import DictConfig

from modelnorm import functional_l2_norm, get_norm_fn, functional_rms_norm
from model.attention import Attention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
	def __init__(self, config: DictConfig):
		super().__init__()
		self.norm_type = config.norm_type
		self.norm_eps = config.norm_eps
		self.use_layer_scale = config.use_layer_scale

		self.attn = Attention(config)
		self.ff = FeedForward(config)
		self.dropout = nn.Dropout(config.dropout)

		if self.norm_type == "f_rmsnorm":
			self.attn_norm = lambda x: functional_rms_norm(x, self.norm_eps)
			self.ff_norm = lambda x: functional_rms_norm(x, self.norm_eps)
		elif self.norm_type == "f_l2norm":
			self.attn_norm = lambda x: functional_l2_norm(x, self.norm_eps)
			self.ff_norm = lambda x: functional_l2_norm(x, self.norm_eps)
		else:
			self.attn_norm = get_norm_fn(config.norm_type, config.d_model, config.norm_eps)
			self.ff_norm = get_norm_fn(config.norm_type, config.d_model, config.norm_eps)

		if self.use_layer_scale:
			self.attn_layer_scale = nn.Parameter(torch.ones(config.d_model))
			self.ff_layer_scale = nn.Parameter(torch.ones(config.d_model))

		# Mark projection layers for residual scaling in _init_weights
		# Handling naming variants (out_proj for your code, c_proj for standard GPT)
		if hasattr(self.attn, "out_proj"):
			self.attn.out_proj.RESIDUAL_SCALE_FLAG = True

		if hasattr(self.ff, "out_proj"):
			self.ff.out_proj.RESIDUAL_SCALE_FLAG = True

	def forward(self, x):
		h_norm = self.attn_norm(x)
		attn_out = self.dropout(self.attn(h_norm))
		if self.use_layer_scale:
			h = x + attn_out * self.attn_layer_scale
		else:
			h = x + attn_out

		x_norm = self.ff_norm(h)
		ff_out = self.dropout(self.ff(x_norm))
		if self.use_layer_scale:
			x = h + ff_out * self.ff_layer_scale
		else:
			x = h + ff_out
		return x
