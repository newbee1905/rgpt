import torch
import torch.nn as nn
from omegaconf import DictConfig
import math

from model.kernels import FusedLinearActivationFunction

# ACTIVATIONS = {
# 	"gelu": F.gelu,
# 	"relu": F.relu,
# 	"silu": F.silu,
# 	"mish": F.mish,
# 	"squared_relu": lambda x: F.relu(x).pow(2),
# }


class FeedForward(nn.Module):
	def __init__(self, config: DictConfig):
		super().__init__()
		self.act_type = config.activation_type

		self.fc1 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
		self.fc2 = nn.Linear(config.d_model * 4, config.d_model, bias=False)

	def forward(self, x):
		bsz, seq_len, d_model = x.shape

		x_2d = x.view(-1, d_model)

		out_2d = FusedLinearActivationFunction.apply(
			x_2d,
			self.fc1.weight,
			self.fc2.weight.t(),
			self.act_type,
		)

		return out_2d.view(bsz, -1, d_model)
