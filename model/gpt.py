import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from model.norm import functional_l2_norm, get_norm_fn, functional_rms_norm
from model.transformer_block import TransformerBlock
from model.utils import trunc_normal_init_


class GPT(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		config = DictConfig(kwargs)
		self.config = config
		self.name = config.name
		self.embs = nn.Embedding(config.vocab_size, config.d_model)

		# No positional embeddings needed for RoPE
		self.dropout = nn.Dropout(config.dropout)
		self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

		self.norm_type = config.norm_type
		self.norm_eps = config.norm_eps
		if self.norm_type == "f_rmsnorm":
			self.ln_f = lambda x: functional_rms_norm(x, self.norm_eps)
		elif self.norm_type == "f_l2norm":
			self.ln_f = lambda x: functional_l2_norm(x, self.norm_eps)
		else:
			self.ln_f = get_norm_fn(config.norm_type, config.d_model, config.norm_eps)

		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

		if hasattr(self.config, "n_recur") and self.config.n_recur > 1:
			self.q_head = nn.Linear(config.d_model, 2, bias=True)
		else:
			self.register_parameter("q_head", None)

		for name, module in self.named_modules():
			self._init_weights(module, name)

	def _init_weights(self, module, name=""):
		if isinstance(module, nn.Linear):
			# Special case for Q-head bootstrapping
			if self.q_head is not None and module is self.q_head:
				with torch.no_grad():
					module.weight.zero_()
					module.bias.fill_(-5.0)
				return

			std = 0.02  # Standard GPT-2 std
			gain = 1.0

			use_muon = module.weight.ndim >= 2 and "embs" not in name and "lm_head" not in name

			if use_muon:
				# Apply residual scaling for depth stability if marked as a residual/projection layer
				if hasattr(module, "RESIDUAL_SCALE_FLAG"):
					# Factor accounts for two residual additions per layer (attn + mlp)
					gain *= (2 * self.config.n_layers) ** -0.5

				# Use higher energy for weights optimized by Muon
				# Standard 0.02 is too small for Newton-Schulz convergence
				nn.init.xavier_uniform_(module.weight, gain=gain)
			else:
				# Apply residual scaling for depth stability if marked as a residual/projection layer
				if hasattr(module, "RESIDUAL_SCALE_FLAG"):
					# Factor accounts for two residual additions per layer (attn + mlp)
					std *= (2 * self.config.n_layers) ** -0.5

				trunc_normal_init_(module.weight, std=std)

			if module.bias is not None:
				nn.init.zeros_(module.bias)

		elif isinstance(module, nn.Embedding):
			trunc_normal_init_(module.weight, std=0.02)

		elif isinstance(module, nn.ParameterList) or isinstance(module, nn.Parameter):
			# This handles layer_scale parameters if they are visited by apply
			pass

		# Manually initialize LayerScale parameters if they exist in the module
		# set the layerscale to 1.0 cause the model is very small
		if hasattr(module, "attn_layer_scale"):
			with torch.no_grad():
				module.attn_layer_scale.fill_(0.1)
		if hasattr(module, "ff_layer_scale"):
			with torch.no_grad():
				module.ff_layer_scale.fill_(0.1)

	def forward(self, input_ids, labels=None):
		bsz, seq_len = input_ids.shape

		x = self.embs(input_ids)
		x = self.dropout(x)

		all_step_q = []
		accum_halt_loss = torch.tensor(0.0, device=x.device)

		for i in range(self.config.n_recur):
			for layer in self.layers:
				x = layer(x)

			if self.config.cal_halt_loss and labels is not None:
				# Current Step Predictions (with Gradients)
				h = self.ln_f(x)

				cur_q = self.q_head(h)  # [halt_logits, continue_logits]
				all_step_q.append(cur_q)

				step_logits = self.lm_head(h)
				step_token_loss = F.cross_entropy(
					step_logits.view(-1, step_logits.size(-1)),
					labels.view(-1),
					ignore_index=self.config.get("pad_token_id", -100),
				)

				with torch.no_grad():
					is_correct = (step_logits.argmax(dim=-1) == labels).float().unsqueeze(-1)

				q_halt_logits = cur_q[..., 0:1]
				accum_halt_loss += F.binary_cross_entropy_with_logits(q_halt_logits, is_correct)

			# Detach for the next recurrence step
			if self.config.detach_recur and i < self.config.n_recur - 1:
				x = x.detach()

		final_h = self.ln_f(x)
		final_logits = self.lm_head(final_h)

		# This is the "lookahead" to compute the Target Q for the continue action (Bellman lookahead)
		target_q_continues = []

		if self.training and self.config.n_recur > 1 and all_step_q:
			with torch.no_grad():
				all_q = torch.stack(all_step_q, dim=0)  # (n_recur, B, L, 2)
				next_q_max = torch.maximum(all_q[1:, ..., 0], all_q[1:, ..., 1])
				last_step_q_halt = all_q[-1:, ..., 0]
				target_q_val = torch.cat([next_q_max, last_step_q_halt], dim=0)
				target_q_continues = torch.sigmoid(target_q_val)

		return final_logits, accum_halt_loss, all_step_q, target_q_continues
