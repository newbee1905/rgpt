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
		 - Logic: hard_scale < 1.0 (e.g., Target=10, Norm=20 -> scale=0.5).
		 - Prevents gradient explosion by projecting onto the hypersphere.

	2. Small Vectors (||x|| < sqrt(d)): Boosted Elastically.
		 - Logic: hard_scale > 1.0 (e.g., Target=10, Norm=5 -> hard_scale=2.0).
		 - Formula: scale = hard_scale^alpha.
		 - Maintains relative magnitude relationships while preventing signal decay.
	"""

	def __init__(self, dim: int, alpha: float = 0.5, eps: float = 1e-6):
		super().__init__()
		self.dim = dim
		self.alpha = alpha
		self.eps = eps

		# Target radius is sqrt(d)
		self.target_radius = dim**0.5

	def forward(self, x: torch.Tensor):
		sq_norm = torch.sum(x.pow(2), dim=-1, keepdim=True)
		inv_norm = torch.rsqrt(sq_norm.clamp(min=self.eps))

		# Calculate the 'Hard' Scale (The RMSNorm Equivalent)
		hard_scale = self.target_radius * inv_norm

		# The Decision Logic:
		# If norm > target: Use Hard Scale (Clamp down to target)
		# If norm < target: Use Elastic Scale (Boost up gently)
		final_scale = torch.where(hard_scale < 1.0, hard_scale, hard_scale.pow(self.alpha))

		return x * final_scale

class ElasticBallNorm(nn.Module):
	def __init__(self, dim: int, alpha: float = 0.5, eps: float = 1e-6):
		"""
		ElasticBallNorm (Universal): Applies a continuous elastic scaling 
		to all vectors regardless of size.

		Args:
			dim: Dimension of the input vectors (d_head).
			alpha: The "Stiffness" of the normalization (0.0 to 1.0).
				 - 1.0 = Rigid RMSNorm (Forces all vectors to sqrt(d)).
				 - 0.5 = Geometric Mean (Pulls vectors halfway to sqrt(d) in log-space).
				 - 0.0 = Identity (No normalization).
			eps: Epsilon for numerical stability.
		"""
		super().__init__()
		self.alpha = alpha
		self.eps = eps
		
		# Target radius corresponds to the expected norm sqrt(d)
		self.target_radius = dim ** 0.5

	def forward(self, x: torch.Tensor):
		norm = x.pow(2).sum(dim=-1, keepdim=True)
		norm = torch.sqrt(norm + self.eps)

		# This represents the multiplier needed to force x EXACTLY to the target.
		# (This is equivalent to the scale factor in RMSNorm)
		ratio = self.target_radius / norm

		# Apply Elasticity
		# Instead of applying the full ratio,
		# we raise it to the power of alpha.
		# - If alpha < 1, we only apply a fraction of the correction.
		scale = ratio.pow(self.alpha)

		return x * scale

class HyperballNorm(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		scale = torch.rsqrt(1.0 + x.pow(2).mean(dim=-1, keepdim=True))
		return x * scale


def get_norm_fn(name, d_model, eps):
	if name == "rmsnorm":
		return RMSNorm(d_model, eps)
	elif name == "layernorm":
		return nn.LayerNorm(d_model, eps)
	elif name == "l2norm":
		return L2Norm(d_model, eps)
	elif name == "elastic":
		return HybridElasticNorm(d_model, eps=eps)
	elif name == "elasticball":
		return ElasticBallNorm(d_model, eps=eps)
	elif name == "hyperball":
		return HyperballNorm()
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
