import math
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
	"""
	Newton–Schulz iteration to compute the "zeroth power" / orthogonalization of G.
	Quintic iteration with coefficients selected to maximize the slope at zero.

	Optimisations:
	  - Scale once (fp32 Fro norm) then run iterations in bf16
	  - Reduce temporaries and use in-place adds where safe
	"""
	assert G.ndim >= 2
	a, b, c = (3.4445, -4.7750, 2.0315)

	X = G.to(dtype=torch.bfloat16)

	transposed = False
	if X.size(-2) > X.size(-1):
		X = X.transpose(-2, -1)
		transposed = True

	X = X.contiguous()

	# Stable one-time scaling in fp32
	scale = X.float().norm(dim=(-2, -1), keepdim=True).clamp_min_(1e-7)
	X = (X / scale).to(dtype=torch.bfloat16)

	for _ in range(steps):
		# A = X X^T
		A = X @ X.transpose(-2, -1)
		A2 = A @ A
		# B = b*A + c*A2
		B = (b * A).add_(A2, alpha=c)
		# X = a*X + B@X
		X = (a * X).add_(B @ X)

	if transposed:
		X = X.transpose(-2, -1)
	return X


# -----------------------------
# Muon update (compiled)
# -----------------------------
@torch.compile(dynamic=False, fullgraph=True)
def muon_update(
	grad_2d: torch.Tensor,
	momentum_2d: torch.Tensor,
	beta: float = 0.95,
	ns_steps: int = 5,
	nesterov: bool = True,
	scale: float = 1.0,
) -> torch.Tensor:
	"""
	Expects grad_2d and momentum_2d to already be 2D views (or at least matching shapes).
	"""
	g = grad_2d.to(dtype=momentum_2d.dtype)

	momentum_2d.lerp_(g, 1.0 - beta)
	if nesterov:
		g = g.lerp_(momentum_2d, beta)
	else:
		g = momentum_2d

	upd = zeropower_via_newtonschulz5(g, steps=ns_steps)
	if scale != 1.0:
		upd.mul_(scale)
	return upd


# -----------------------------
# AdamW update (foreach)
# -----------------------------
def foreach_adamw_step(
	params: List[torch.Tensor],
	grads: List[torch.Tensor],
	exp_avgs: List[torch.Tensor],
	exp_avg_sqs: List[torch.Tensor],
	step: int,
	lr: float,
	wd: float,
	beta1: float,
	beta2: float,
	eps: float,
) -> None:
	"""
	Low-overhead AdamW using foreach ops.
	"""
	if wd != 0.0:
		torch._foreach_mul_(params, 1.0 - lr * wd)

	# m = beta1*m + (1-beta1)*g
	torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)

	# v = beta2*v + (1-beta2)*g^2
	grad_sq = torch._foreach_mul(grads, grads)
	torch._foreach_lerp_(exp_avg_sqs, grad_sq, 1.0 - beta2)

	bc1 = 1.0 - (beta1**step)
	bc2 = 1.0 - (beta2**step)
	inv_bc1 = 1.0 / bc1
	inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)

	denom = torch._foreach_sqrt(exp_avg_sqs)
	torch._foreach_mul_(denom, inv_sqrt_bc2)
	torch._foreach_add_(denom, eps)

	step_updates = torch._foreach_div(exp_avgs, denom)
	torch._foreach_mul_(step_updates, inv_bc1)
	torch._foreach_add_(params, step_updates, alpha=-lr)


# -----------------------------
# Helpers
# -----------------------------
def _as_2d_view(t: torch.Tensor) -> torch.Tensor:
	# For >=2D tensors, treat first dim as "rows" and flatten the rest.
	# This covers linear weights, conv kernels, etc.
	if t.ndim >= 2:
		return t.view(t.size(0), -1)
	return t


def _muon_scale(m: int, n: int) -> float:
	# sqrt(max(1, m/n))
	r = m / n
	return math.sqrt(r) if r > 1.0 else 1.0


def _default_device_like(p: torch.Tensor) -> torch.device:
	return p.device


# -----------------------------
# Single-device Muon
# -----------------------------
class Muon(torch.optim.Optimizer):
	"""
	Muon - MomentUm Orthogonalized by Newton–Schulz

	Applies Muon to "hidden weights" (2D+) and AdamW to others (biases, norms, embeddings if desired).
	You control this via param groups: set group["use_muon"]=True/False.

	Speed improvements:
	  - foreach AdamW for aux params
	  - compiled Muon update and NS
	  - avoid per-step shape logic inside compiled graphs
	  - group-level AdamW step counter (lower Python overhead)
	"""

	def __init__(
		self,
		params: List[Dict[str, Any]],
		lr: float = 3e-4,
		weight_decay: float = 0.01,
		momentum: float = 0.95,
		nesterov: bool = True,
		adam_betas: Tuple[float, float] = (0.9, 0.95),
		adam_eps: float = 1e-8,
		muon_lr_multiplier: float = 100.0,
		ns_steps: int = 5,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
			muon_lr_multiplier=muon_lr_multiplier,
			ns_steps=ns_steps,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure: Optional[Any] = None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			use_muon = group.get("use_muon", False)
			lr = float(group["lr"])
			wd = float(group["weight_decay"])

			if use_muon:
				muon_lr = lr * float(group["muon_lr_multiplier"])
				beta = float(group["momentum"])
				nesterov = bool(group["nesterov"])
				ns_steps = int(group.get("ns_steps", 5))

				for p in group["params"]:
					g = p.grad
					if g is None:
						continue

					st = self.state[p]
					if len(st) == 0:
						# Keep momentum in bf16 to reduce bandwidth; it works well for this use.
						st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)

					if wd != 0.0:
						p.mul_(1.0 - lr * wd)

					g2d = _as_2d_view(g)
					m2d = _as_2d_view(st["momentum_buffer"])

					scale = 1.0
					if g2d.ndim == 2:
						scale = _muon_scale(g2d.size(0), g2d.size(1))

					upd2d = muon_update(
						g2d,
						m2d,
						beta=beta,
						ns_steps=ns_steps,
						nesterov=nesterov,
						scale=scale,
					)
					p.add_(upd2d.view_as(p), alpha=-muon_lr)

			else:
				# foreach AdamW path
				beta1, beta2 = group["adam_betas"]
				eps = float(group["adam_eps"])

				# group-level step counter (much cheaper than per param)
				step_i = int(group.get("_adam_step", 0)) + 1
				group["_adam_step"] = step_i

				ps: List[torch.Tensor] = []
				gs: List[torch.Tensor] = []
				m1s: List[torch.Tensor] = []
				m2s: List[torch.Tensor] = []

				for p in group["params"]:
					g = p.grad
					if g is None:
						continue

					st = self.state[p]
					if len(st) == 0:
						st["exp_avg"] = torch.zeros_like(p)
						st["exp_avg_sq"] = torch.zeros_like(p)

					ps.append(p)
					gs.append(g)
					m1s.append(st["exp_avg"])
					m2s.append(st["exp_avg_sq"])

				if ps:
					foreach_adamw_step(
						ps,
						gs,
						m1s,
						m2s,
						step=step_i,
						lr=lr,
						wd=wd,
						beta1=float(beta1),
						beta2=float(beta2),
						eps=eps,
					)

		return loss


# -----------------------------
# Distributed Muon (DDP version)
# -----------------------------
class DistMuon(torch.optim.Optimizer):
	"""
	Distributed Muon for DDP-style training.

	IMPORTANT:
	  If you are using PyTorch DDP, gradients are already synchronized (all-reduced) during backward.
	  That means every rank has the same grads, so the fastest approach is to run the *same* optimizer
	  step locally on each rank with **no extra communication**.

	This class does exactly that. It is typically MUCH faster than optimizer sharding schemes that
	broadcast parameters/updates every step.

	If you are not using DDP (or not syncing grads), you must add your own gradient synchronization.
	"""

	def __init__(
		self,
		params: List[Dict[str, Any]],
		lr: float = 3e-4,
		weight_decay: float = 0.01,
		momentum: float = 0.95,
		nesterov: bool = True,
		adam_betas: Tuple[float, float] = (0.9, 0.95),
		adam_eps: float = 1e-8,
		muon_lr_multiplier: float = 100.0,
		ns_steps: int = 5,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
			muon_lr_multiplier=muon_lr_multiplier,
			ns_steps=ns_steps,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure: Optional[Any] = None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		# Optional sanity check (cheap): if dist not initialised, behave like normal.
		dist_ready = dist.is_available() and dist.is_initialized()

		for group in self.param_groups:
			use_muon = group.get("use_muon", False)
			lr = float(group["lr"])
			wd = float(group["weight_decay"])

			if use_muon:
				muon_lr = lr * float(group["muon_lr_multiplier"])
				beta = float(group["momentum"])
				nesterov = bool(group["nesterov"])
				ns_steps = int(group.get("ns_steps", 5))

				for p in group["params"]:
					g = p.grad
					if g is None:
						continue

					# If you're NOT using DDP gradient sync, you could sync here.
					# But for DDP, this would be redundant and slow.
					# if dist_ready: dist.all_reduce(g, op=dist.ReduceOp.SUM); g.div_(dist.get_world_size())

					st = self.state[p]
					if len(st) == 0:
						st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)

					if wd != 0.0:
						p.mul_(1.0 - lr * wd)

					g2d = _as_2d_view(g)
					m2d = _as_2d_view(st["momentum_buffer"])

					scale = 1.0
					if g2d.ndim == 2:
						scale = _muon_scale(g2d.size(0), g2d.size(1))

					upd2d = muon_update(
						g2d,
						m2d,
						beta=beta,
						ns_steps=ns_steps,
						nesterov=nesterov,
						scale=scale,
					)
					p.add_(upd2d.view_as(p), alpha=-muon_lr)

			else:
				beta1, beta2 = group["adam_betas"]
				eps = float(group["adam_eps"])

				step_i = int(group.get("_adam_step", 0)) + 1
				group["_adam_step"] = step_i

				ps: List[torch.Tensor] = []
				gs: List[torch.Tensor] = []
				m1s: List[torch.Tensor] = []
				m2s: List[torch.Tensor] = []

				for p in group["params"]:
					g = p.grad
					if g is None:
						continue

					# Same note: DDP already syncs grads.
					# if dist_ready: dist.all_reduce(g, op=dist.ReduceOp.SUM); g.div_(dist.get_world_size())

					st = self.state[p]
					if len(st) == 0:
						st["exp_avg"] = torch.zeros_like(p)
						st["exp_avg_sq"] = torch.zeros_like(p)

					ps.append(p)
					gs.append(g)
					m1s.append(st["exp_avg"])
					m2s.append(st["exp_avg_sq"])

				if ps:
					foreach_adamw_step(
						ps,
						gs,
						m1s,
						m2s,
						step=step_i,
						lr=lr,
						wd=wd,
						beta1=float(beta1),
						beta2=float(beta2),
						eps=eps,
					)

		return loss
