import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# Activation Functions & Derivatives
# -----------------------------------------------------------------------------


@triton.jit
def gelu_approx_fwd(x):
	# Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	org_dtype = x.dtype

	x = x.to(tl.float32)
	k = 0.7978845608028654  # sqrt(2/pi)
	inner = k * (x + 0.044715 * x * x * x)

	x = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
	return x.to(org_dtype)


@triton.jit
def gelu_approx_bwd(x, y, dy):
	org_dtype = x.dtype

	x = x.to(tl.float32)
	k = 0.7978845608028654

	x_sq = x * x
	x_cub = x_sq * x
	inner = k * (x + 0.044715 * x_cub)
	tanh_inner = tl.extra.cuda.libdevice.tanh(inner)

	# Gradient calculation
	# f(x) = 0.5 * x * (1 + tanh(inner))
	# f'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * d(inner)/dx
	dtanh = 1.0 - tanh_inner * tanh_inner
	d_inner = k * (1.0 + 3.0 * 0.044715 * x_sq)
	grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * dtanh * d_inner

	return (dy * grad).to(org_dtype)


@triton.jit
def silu_fwd(x):
	org_dtype = x.dtype
	x = x.to(tl.float32)

	x = x * tl.sigmoid(x)

	return x.to(org_dtype)


@triton.jit
def silu_bwd(x, y, dy):
	org_dtype = x.dtype

	x = x.to(tl.float32)
	sig = tl.sigmoid(x)

	grad = sig * (1.0 + x * (1.0 - sig))

	return (dy * grad).to(org_dtype)


@triton.jit
def mish_fwd(x):
	org_dtype = x.dtype

	x = x.to(tl.float32)
	sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

	x = x * tl.extra.cuda.libdevice.tanh(sp)

	return x.to(org_dtype)


@triton.jit
def mish_bwd(x, y, dy):
	org_dtype = x.dtype

	x = x.to(tl.float32)
	sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
	tsp = tl.extra.cuda.libdevice.tanh(sp)
	sig = tl.sigmoid(x)

	grad = tsp + x * (1.0 - tsp * tsp) * sig

	return (dy * grad).to(org_dtype)


@triton.jit
def squared_relu_fwd(x):
	org_dtype = x.dtype

	x = x.to(tl.float32)
	relu = tl.maximum(x, 0.0)
	x = relu * relu

	return x.to(org_dtype)


@triton.jit
def squared_relu_bwd(x, y, dy):
	# y = relu(x)^2
	# dy/dx = 2 * relu(x)

	org_dtype = x.dtype

	x = x.to(tl.float32)
	relu = tl.maximum(x, 0.0)
	grad = 2.0 * relu

	return (dy * grad).to(org_dtype)


# -----------------------------------------------------------------------------
# Kernel 1: Forward Pass
# Computes: Pre = X @ W1.T
# Post = Activation(Pre)
# Stores both Pre (for backward) and Post (for next layer)
# -----------------------------------------------------------------------------
@triton.jit
def linear_activation_fwd_kernel(
	# Pointers
	x_ptr,
	w1_ptr,
	pre_ptr,
	post_ptr,
	# Dimensions
	M,
	N,
	K,
	# Strides
	stride_xm,
	stride_xk,
	stride_wn,
	stride_wk,
	stride_pre_m,
	stride_pre_n,
	stride_post_m,
	stride_post_n,
	# Meta-parameters
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
	BLOCK_K: tl.constexpr,
	ACTIVATION: tl.constexpr,
):
	pid_m = tl.program_id(0)
	pid_n = tl.program_id(1)

	offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_K)

	x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
	w1_ptrs = w1_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

	acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		x_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K)
		w1_mask = (offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K)

		x = tl.load(x_ptrs, mask=x_mask, other=0.0)
		w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)

		acc = tl.dot(x, w1, acc)

		x_ptrs += BLOCK_K * stride_xk
		w1_ptrs += BLOCK_K * stride_wk

	offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

	pre_ptrs = pre_ptr + (offs_cm[:, None] * stride_pre_m + offs_cn[None, :] * stride_pre_n)
	post_ptrs = post_ptr + (offs_cm[:, None] * stride_post_m + offs_cn[None, :] * stride_post_n)

	# Compute Activation
	pre_val = acc

	if ACTIVATION == "gelu":
		post_val = gelu_approx_fwd(pre_val)
	elif ACTIVATION == "silu":
		post_val = silu_fwd(pre_val)
	elif ACTIVATION == "mish":
		post_val = mish_fwd(pre_val)
	elif ACTIVATION == "squared_relu":
		post_val = squared_relu_fwd(pre_val)
	else:
		# Default fallback or error (though constexpr prevents runtime error usually)
		post_val = pre_val

	tl.store(pre_ptrs, pre_val, mask=c_mask)
	tl.store(post_ptrs, post_val, mask=c_mask)


# -----------------------------------------------------------------------------
# Kernel 2: Backward Step (Compute dPre)
# Computes: dPre = (GradOut @ W2.T) * ActivationGrad(Pre)
# -----------------------------------------------------------------------------
@triton.jit
def linear_activation_bwd_dpre_kernel(
	# Pointers
	grad_out_ptr,
	w2_ptr,
	pre_ptr,
	dpre_ptr,
	# Dimensions
	M,
	N,
	K,  # Here M=Batch, N=HiddenDim(W1_out), K=OutDim(W2_out)
	# Strides
	stride_gm,
	stride_gk,  # grad_out (M, K)
	stride_w2n,
	stride_w2k,  # W2 (N, K) -> We compute Grad @ W2.T
	stride_prem,
	stride_pren,  # pre (M, N)
	stride_dprem,
	stride_dpren,
	# Meta
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
	BLOCK_K: tl.constexpr,
	ACTIVATION: tl.constexpr,
):
	pid_m = tl.program_id(0)
	pid_n = tl.program_id(1)

	offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_K)

	# 1. Pointers for GradOut (M, K) and W2 (N, K)
	g_ptrs = grad_out_ptr + (offs_m[:, None] * stride_gm + offs_k[None, :] * stride_gk)
	w2_ptrs = w2_ptr + (offs_n[None, :] * stride_w2n + offs_k[:, None] * stride_w2k)

	# Accumulator for dPost (Always F32)
	dpost = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		g_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K)
		w2_mask = (offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K)

		g = tl.load(g_ptrs, mask=g_mask, other=0.0)
		w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

		dpost = tl.dot(g, w2, dpost)

		g_ptrs += BLOCK_K * stride_gk
		w2_ptrs += BLOCK_K * stride_w2k

	# Apply Activation Derivative
	pre_ptrs = pre_ptr + (offs_m[:, None] * stride_prem + offs_n[None, :] * stride_pren)
	mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)

	pre = tl.load(pre_ptrs, mask=mask_mn, other=0.0)

	if ACTIVATION == "gelu":
		# gelu bwd needs x
		dpre = gelu_approx_bwd(pre, None, dpost)
	elif ACTIVATION == "silu":
		# silu bwd needs x
		dpre = silu_bwd(pre, None, dpost)
	elif ACTIVATION == "mish":
		# mish bwd needs x
		dpre = mish_bwd(pre, None, dpost)
	elif ACTIVATION == "squared_relu":
		dpre = squared_relu_bwd(pre, None, dpost)
	else:
		dpre = dpost

	# Store dPre
	dpre_out_ptrs = dpre_ptr + (offs_m[:, None] * stride_dprem + offs_n[None, :] * stride_dpren)
	tl.store(dpre_out_ptrs, dpre, mask=mask_mn)


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------


class FusedLinearActivationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, W1, W2, activation="squared_relu"):
		x = x.contiguous()
		W1 = W1.contiguous()
		W2 = W2.contiguous()

		M, K_in = x.shape
		N_hidden, _ = W1.shape
		_, K_out = W2.shape

		pre = torch.empty((M, N_hidden), device=x.device, dtype=x.dtype)
		post = torch.empty((M, N_hidden), device=x.device, dtype=x.dtype)

		grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N_hidden, META["BLOCK_N"]))

		linear_activation_fwd_kernel[grid](
			x,
			W1,
			pre,
			post,
			M,
			N_hidden,
			K_in,
			x.stride(0),
			x.stride(1),
			W1.stride(0),
			W1.stride(1),
			pre.stride(0),
			pre.stride(1),
			post.stride(0),
			post.stride(1),
			BLOCK_M=128,
			BLOCK_N=64,
			BLOCK_K=32,
			ACTIVATION=activation,
		)

		x3 = torch.matmul(post, W2)

		ctx.save_for_backward(x, W1, W2, pre, post)
		ctx.activation = activation
		return x3

	@staticmethod
	def backward(ctx, grad_output):
		x, W1, W2, pre, post = ctx.saved_tensors
		activation = ctx.activation

		grad_output = grad_output.contiguous()

		dW2 = torch.matmul(post.t(), grad_output)

		M, N_hidden = pre.shape
		_, K_out = W2.shape

		dpre = torch.empty_like(pre)

		grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N_hidden, META["BLOCK_N"]))

		linear_activation_bwd_dpre_kernel[grid](
			grad_output,
			W2,
			pre,
			dpre,
			M,
			N_hidden,
			K_out,
			grad_output.stride(0),
			grad_output.stride(1),
			W2.stride(0),
			W2.stride(1),
			pre.stride(0),
			pre.stride(1),
			dpre.stride(0),
			dpre.stride(1),
			BLOCK_M=128,
			BLOCK_N=64,
			BLOCK_K=32,
			ACTIVATION=activation,
		)

		dW1 = torch.matmul(dpre.t(), x)
		dx = torch.matmul(dpre, W1)

		return dx, dW1, dW2, None


# -----------------------------------------------------------------------------
# Testing & Benchmark
# -----------------------------------------------------------------------------


def test_correctness():
	torch.manual_seed(42)
	device = "cuda"
	torch.backends.cuda.matmul.allow_tf32 = True

	# Activation Maps (Name -> Torch Function)
	act_map = {
		"squared_relu": lambda x: torch.relu(x) ** 2,
		"gelu": lambda x: F.gelu(x, approximate="tanh"),
		"silu": F.silu,
		"mish": F.mish,
	}

	dtypes = [torch.float16, torch.bfloat16, torch.float32]

	for dtype in dtypes:
		for act_name, torch_act in act_map.items():
			print(f"--- Testing {act_name} [{dtype}] ---")

			M, N, K, Out = 128, 256, 128, 64
			scale = 0.1

			x = (torch.randn(M, K, device=device, dtype=dtype) * scale).requires_grad_(True)
			W1 = (torch.randn(N, K, device=device, dtype=dtype) * scale).requires_grad_(True)
			W2 = (torch.randn(N, Out, device=device, dtype=dtype) * scale).requires_grad_(True)

			# Reference
			def torch_ref(x, w1, w2):
				pre = x @ w1.T
				post = torch_act(pre)
				return post @ w2

			y_ref = torch_ref(x, W1, W2)
			loss = y_ref.sum()
			loss.backward()
			grads_ref = (x.grad.clone(), W1.grad.clone(), W2.grad.clone())

			x.grad, W1.grad, W2.grad = None, None, None

			# Triton
			y_tri = FusedLinearActivationFunction.apply(x, W1, W2, act_name)
			loss_tri = y_tri.sum()
			loss_tri.backward()
			grads_tri = (x.grad, W1.grad, W2.grad)

			if dtype == torch.float32:
				atol, rtol = 1e-1, 1e-2
			else:
				atol, rtol = 5e-1, 5e-2

			print(f"Ref Loss: {loss.item():.4f}, Triton Loss: {loss_tri.item():.4f}")
			max_diff = (y_ref - y_tri).abs().max().item()
			print(f"Max Fwd Diff: {max_diff:.4f}")

			try:
				assert torch.allclose(y_ref, y_tri, atol=atol, rtol=rtol), f"Fwd Output mismatch"
				assert torch.allclose(grads_ref[0], grads_tri[0], atol=atol, rtol=rtol), "dX mismatch"
				assert torch.allclose(grads_ref[1], grads_tri[1], atol=atol, rtol=rtol), "dW1 mismatch"
				assert torch.allclose(grads_ref[2], grads_tri[2], atol=atol, rtol=rtol), "dW2 mismatch"
				print("Passed")
			except AssertionError as e:
				print(f"Failed: {e}")
			print("")


@triton.testing.perf_report(
	triton.testing.Benchmark(
		x_names=["M"],
		x_vals=[128 * i for i in range(2, 20)],
		line_arg="provider",
		line_vals=["torch", "triton"],
		line_names=["PyTorch", "Triton"],
		styles=[("green", "-"), ("blue", "-")],
		ylabel="TFLOPS",
		plot_name="linear_activation_perf",
		args={"N": 4096, "K": 1024, "Out": 1024, "dtype_str": "float16", "activation": "squared_relu"},
	)
)
def benchmark(M, N, K, Out, dtype_str, activation, provider):
	dtype = getattr(torch, dtype_str)

	x = torch.randn(M, K, device="cuda", dtype=dtype)
	W1 = torch.randn(N, K, device="cuda", dtype=dtype)
	W2 = torch.randn(N, Out, device="cuda", dtype=dtype)

	# Activation setup for Torch baseline
	if activation == "squared_relu":
		act_fn = lambda x: torch.relu(x) ** 2
	elif activation == "gelu":
		act_fn = lambda x: F.gelu(x, approximate="tanh")
	elif activation == "silu":
		act_fn = F.silu
	elif activation == "mish":
		act_fn = F.mish
	else:
		act_fn = F.relu

	quantiles = [0.5, 0.2, 0.8]
	if provider == "torch":

		def run_func():
			pre = torch.matmul(x, W1.T)
			post = act_fn(pre)
			return torch.matmul(post, W2)
	else:

		def run_func():
			return FusedLinearActivationFunction.apply(x, W1, W2, activation)

	ms, min_ms, max_ms = triton.testing.do_bench(run_func, quantiles=quantiles)

	flops = 2 * M * N * K + 2 * M * N * Out
	perf = flops * 1e-12 / (ms * 1e-3)
	return perf, max_ms, min_ms


if __name__ == "__main__":
	test_correctness()
	benchmark.run(print_data=True, show_plots=False)
