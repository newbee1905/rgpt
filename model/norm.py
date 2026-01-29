import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class RMSNorm(nn.Module):
	def __init__(
		self,
		dim: int,
		eps: float = 1e-6,
		add_unit_offset: bool = True,
	):
		super().__init__()

		self.register_buffer("eps", torch.tensor(float(eps)))
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def forward(self, x):
		# Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
		# See https://github.com/huggingface/transformers/pull/29402

		variance = x.pow(2).mean(dim=-1, keepdim=True)
		out = x * torch.rsqrt(variance + self.eps)
		out = out.to(self.weight.dtype)

		if self.add_unit_offset:
			out = out * (1 + self.weight)
		else:
			out = out * self.weight

		return out


def functional_rms_norm(x, eps=1e-5):
	return F.rms_norm(x, (x.size(-1),))


class L2Norm(nn.Module):
	def __init__(self, d_model, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.temp = nn.Parameter(torch.ones(1))

	def forward(self, x):
		return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + self.eps) * self.temp


def functional_l2_norm(x, eps=1e-5):
	return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + eps)


class HybridElasticNorm(nn.Module):
	"""
	A Parameter-Free Norm that treats 'Signal' (Large) and 'Noise/Weak' (Small)
	vectors differently to optimize Attention dynamics.

	Logic:
	1. Large Vectors (||x|| > sqrt(d)): Clamped rigidly to the surface.
	   - Prevents gradient explosion.
	   - Gradient behavior: Projection (Stable).

	2. Small Vectors (||x|| < sqrt(d)): Boosted Elastically.
	   - Formula: scale = (Target / Input)^alpha
	   - Instead of forcing them to the surface (RMSNorm), we gently pull
		 them closer, maintaining their relative order.
	   - Gradient behavior: Boosted, but strictly safer than RMSNorm.
	"""

	def __init__(self, dim: int, alpha: float = 0.5, eps: float = 1e-6):
		super().__init__()
		self.dim = dim
		self.alpha = alpha
		self.eps = eps

		# Target radius is sqrt(d)
		self.target_radius = dim**0.5

	def forward(self, x: torch.Tensor):
		# Calculate the raw magnitude of input vectors
		norm = x.norm(p=2, dim=-1, keepdim=True)
		safe_norm = torch.clamp(norm, min=self.eps)

		# Calculate the 'Hard' Scale (The RMSNorm Equivalent)
		hard_scale = self.target_radius / safe_norm

		# Calculate the 'Elastic' Scale
		elastic_scale = hard_scale.pow(self.alpha)

		# The Decision Logic:
		# If norm > target: Use Hard Scale (Clamp down to target)
		# If norm < target: Use Elastic Scale (Boost up gently)
		final_scale = torch.minimum(hard_scale, elastic_scale)

		return x * final_scale

def get_norm_fn(name, d_model, eps):
	if name == "rmsnorm":
		return RMSNorm(d_model, eps)
	elif name == "layernorm":
		return nn.LayerNorm(d_model, eps)
	elif name == "l2norm":
		return L2Norm(d_model, eps)
	elif name == "elastic":
		return HybridElasticNorm(d_model, eps=eps)
	else:
		return nn.Identity()

class QKNorm(nn.Module):
	def __init__(self, d_head, norm_type, norm_eps):
		super().__init__()
		if norm_type == "f_rmsnorm":
			self.q_norm_fn = lambda q: functional_rms_norm(q, norm_eps)
			self.k_norm_fn = lambda k: functional_rms_norm(k, norm_eps)
		elif norm_type == "f_l2norm":
			self.q_norm_fn = lambda q: functional_l2_norm(q, norm_eps)
			self.k_norm_fn = lambda k: functional_l2_norm(k, norm_eps)
		else:
			self.q_norm_fn = get_norm_fn(norm_type, d_head, norm_eps)
			self.k_norm_fn = get_norm_fn(norm_type, d_head, norm_eps)

	def forward(self, q, k):
		return self.q_norm_fn(q), self.k_norm_fn(k)
