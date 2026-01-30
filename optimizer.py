import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
	"""
	Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
	Quintic iteration with coefficients selected to maximize the slope at zero.
	"""
	assert G.ndim >= 2
	a, b, c = (3.4445, -4.7750, 2.0315)
	X = G.bfloat16()
	if G.size(-2) > G.size(-1):
		X = X.mT

	# Ensure spectral norm is at most 1
	X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

	# Perform the NS iterations
	for _ in range(steps):
		A = X @ X.mT
		B = b * A + c * A @ A
		X = a * X + B @ X

	if G.size(-2) > G.size(-1):
		X = X.mT
	return X


@torch.compile(dynamic=False, fullgraph=True)
def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta=0.95, ns_steps=5, nesterov=True):
	"""Performs the Muon update step using momentum and Newton-Schulz orthogonalization."""
	momentum.lerp_(grad, 1 - beta)
	update = grad.lerp_(momentum, beta) if nesterov else momentum

	# Handle convolutional filters by flattening trailing dimensions
	if update.ndim == 4:
		update = update.view(len(update), -1)

	update = zeropower_via_newtonschulz5(update, steps=ns_steps)
	# Scale by sqrt(max(rows, cols)) to maintain variance
	update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
	return update


def adamw_update(
	grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, step: int, betas: tuple, eps: float
):
	"""Performs a standard AdamW update step."""
	exp_avg.lerp_(grad, 1 - betas[0])
	exp_avg_sq.lerp_(grad.square(), 1 - betas[1])

	bias_correction1 = 1 - betas[0] ** step
	bias_correction2 = 1 - betas[1] ** step

	denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
	return (exp_avg / bias_correction1) / denom


class Muon(torch.optim.Optimizer):
	"""
	Muon - MomentUm Orthogonalized by Newton-schulz (Single Device)

	This optimizer automatically applies Muon to hidden weights (2D+) and AdamW to
	others (embeddings, biases, scalars).

	Args:
	    params: List of param groups or parameters.
	    lr: Base learning rate (applied to AdamW parts).
	    weight_decay: AdamW-style weight decay.
	    momentum: Momentum for Muon (beta1 for AdamW).
	    nesterov: Whether to use Nesterov momentum for Muon.
	    adam_betas: Betas for the auxiliary AdamW.
	    adam_eps: Epsilon for the auxiliary AdamW.
	    muon_lr_multiplier: Multiplier for Muon LR relative to base LR.
	"""

	def __init__(
		self,
		params: List[Dict[str, Any]],
		lr=3e-4,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		adam_betas=(0.9, 0.95),
		adam_eps=1e-8,
		muon_lr_multiplier=100.0,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
			muon_lr_multiplier=muon_lr_multiplier,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			use_muon = group.get("use_muon", False)
			lr = group["lr"]
			wd = group["weight_decay"]

			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				if len(state) == 0:
					state["step"] = 0
					if use_muon:
						state["momentum_buffer"] = torch.zeros_like(p)
					else:
						state["exp_avg"] = torch.zeros_like(p)
						state["exp_avg_sq"] = torch.zeros_like(p)

				state["step"] += 1

				# Apply weight decay
				if wd != 0:
					p.mul_(1 - lr * wd)

				if use_muon:
					# Muon update
					muon_lr = lr * group["muon_lr_multiplier"]
					update = muon_update(
						p.grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"]
					)
					p.add_(update.reshape(p.shape), alpha=-muon_lr)
				else:
					# AdamW update
					update = adamw_update(
						p.grad,
						state["exp_avg"],
						state["exp_avg_sq"],
						state["step"],
						group["adam_betas"],
						group["adam_eps"],
					)
					p.add_(update, alpha=-lr)
		return loss


class DistMuon(torch.optim.Optimizer):
	"""
	Distributed version of Muon.
	Uses all_gather to synchronize updates across the world size, processing parameters in shards.
	"""

	def __init__(
		self,
		params: List[Dict[str, Any]],
		lr=3e-4,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		adam_betas=(0.9, 0.95),
		adam_eps=1e-8,
		muon_lr_multiplier=100.0,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
			muon_lr_multiplier=muon_lr_multiplier,
		)

		# Sort Muon parameters by size for better load balancing during sharding
		for group in params:
			if group.get("use_muon", False):
				group["params"] = sorted(group["params"], key=lambda x: x.numel(), reverse=True)

		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		world_size = dist.get_world_size()
		rank = dist.get_rank()

		for group in self.param_groups:
			use_muon = group.get("use_muon", False)
			lr = group["lr"]
			wd = group["weight_decay"]

			if use_muon:
				params = group["params"]
				# Pad params list to be divisible by world_size for even sharding
				padding = (world_size - len(params) % world_size) % world_size
				params_pad = params + [torch.empty_like(params[-1])] * padding

				for base_i in range(0, len(params_pad), world_size):
					# Each rank processes one parameter in this block
					p_idx = base_i + rank
					if p_idx < len(params):
						p = params[p_idx]
						if p.grad is None:
							p.grad = torch.zeros_like(p)

						state = self.state[p]
						if len(state) == 0:
							state["momentum_buffer"] = torch.zeros_like(p)

						if wd != 0:
							p.mul_(1 - lr * wd)

						muon_lr = lr * group["muon_lr_multiplier"]
						update = muon_update(
							p.grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"]
						)
						p.add_(update.reshape(p.shape), alpha=-muon_lr)

					# Sync updates across all ranks
					dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + rank])
			else:
				# Standard non-distributed AdamW for aux parameters
				for p in group["params"]:
					if p.grad is None:
						continue
					state = self.state[p]
					if len(state) == 0:
						state["step"] = 0
						state["exp_avg"] = torch.zeros_like(p)
						state["exp_avg_sq"] = torch.zeros_like(p)

					state["step"] += 1
					if wd != 0:
						p.mul_(1 - lr * wd)

					update = adamw_update(
						p.grad,
						state["exp_avg"],
						state["exp_avg_sq"],
						state["step"],
						group["adam_betas"],
						group["adam_eps"],
					)
					p.add_(update, alpha=-lr)

		return loss
