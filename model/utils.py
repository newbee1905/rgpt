import torch
import math


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
	"""
	Mathematically correct truncated normal initialization.
	PyTorch version of jax.random.truncated_normal (default in Flax).
	Ensures the standard deviation of the initialized tensor is actually 'std'.
	"""
	with torch.no_grad():
		if std == 0:
			tensor.zero_()
		else:
			sqrt2 = math.sqrt(2)
			# Probability density at boundaries
			a = math.erf(lower / sqrt2)
			b = math.erf(upper / sqrt2)
			z = (b - a) / 2

			# Correction factor to ensure final std matches target std
			c = (2 * math.pi) ** -0.5
			pdf_u = c * math.exp(-0.5 * lower**2)
			pdf_l = c * math.exp(-0.5 * upper**2)

			# Formula for variance of truncated normal distribution
			# Var = 1 - (u*pdf(u) - l*pdf(l))/Z - ((pdf(u)-pdf(l))/Z)^2
			var_correction = 1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
			comp_std = std / math.sqrt(var_correction)

			# Inverse CDF sampling
			tensor.uniform_(a, b)
			tensor.erfinv_()
			tensor.mul_(sqrt2 * comp_std)
			tensor.clip_(lower * comp_std, upper * comp_std)

	return tensor
