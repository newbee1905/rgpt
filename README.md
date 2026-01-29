# rGPT: A Recursive Transformer

This repository contains the code for rGPT, a research framework for training both standard and recursive GPT-style transformer models. The project is built with PyTorch and leverages Hydra for flexible configuration management and `torch.distributed` for large-scale training.

It features a custom `NorMuon` optimizer, advanced normalization techniques, and custom high-performance Triton kernels.

## Features

- **Recursive Transformer Architecture**: Implements a recurrent GPT (`rGPT`) where transformer blocks can be applied multiple times to the same sequence, controlled by an adaptive halting mechanism.
- **Standard GPT Model**: Also supports training standard decoder-only transformers.
- **Distributed Training**: Full support for Distributed Data Parallel (DDP) training, with a custom distributed optimizer (`DistNorMuon`).
- **Advanced Optimizer (`NorMuon`)**: A hybrid optimizer combining AdamW for standard parameters (biases, norms) and a novel gradient-on-manifold approach for weight matrices, which uses Newton-Schulz iteration (`polar_express`) to keep weights semi-orthogonal.
- **High-Performance Kernels**: Includes custom Triton kernels for fused linear layers and activation functions (`Squared ReLU`, `GELU`, `SiLU`, `Mish`), improving training speed.
- **Advanced Normalization**:
	- **QK Norm**: Applies normalization to Query and Key vectors within the attention mechanism.
	- **Multiple Norm Layers**: Supports `RMSNorm`, `LayerNorm`, `L2Norm`, and a custom `HybridElasticNorm`.
- **Rotary Position Embeddings (RoPE)**: Modern and efficient relative position encoding.
- **Comprehensive Configuration**: Uses [Hydra](https://hydra.cc/) for managing all experiment parameters, allowing easy modification and composability via YAML files and command-line overrides.
- **Efficient Training**: Employs `bfloat16` mixed-precision, gradient accumulation, and model compilation (`torch.compile`).

## Architecture

The project is structured to separate concerns, making it modular and extensible.

- **`main.py`**: The main entry point for training. It handles DDP setup, dataset/dataloader creation, model instantiation, and orchestrates the training process via the `Trainer`.
- **`trainer.py`**: The core training and evaluation loop. It manages the optimizer, learning rate scheduler, gradient scaling, checkpointing, and logging. It also contains the specialized loss function for the recursive model.
- **`model/`**: Contains all neural network modules.
	- `gpt.py`: The main `GPT` class, which implements the optional recurrence loop and the adaptive halting head (`q_head`).
	- `transformer_block.py`: A standard pre-norm transformer block.
	- `attention.py`: Multi-head self-attention with RoPE and optional Gated Attention (`g1_gate`).
	- `feed_forward.py`: A fused feed-forward network that utilizes the custom Triton kernel.
	- `kernels.py`: Custom Triton kernels for high-performance fused operations.
- **`optimizer.py`**: Implementation of the `NorMuon` and `DistNorMuon` optimizers. This is a key component of the project.
- **`norms.py`**: Custom normalization layer implementations (`RMSNorm`, `L2Norm`, `HybridElasticNorm`).
- **`rope.py`**: Implementation of Rotary Position Embeddings.
- **`conf/`**: The Hydra configuration directory.
	- `config.yaml`: The main configuration file.
	- `model/`, `dataset/`, `training/`: Subdirectories for organizing model architecture, dataset specifics, and training parameters.

## Key Components

### Recursive Model (`rGPT`)

The recursive capability is enabled by setting `model.n_recur > 1` in the configuration. The `GPT` model in `model/gpt.py` will loop over the transformer layers `n_recur` times.

- **Adaptive Halting**: To control the computational cost, the model includes a `q_head` which predicts a "halting" probability at each recurrence step. The loss function (`trainer.py:compute_loss`) includes terms to train this head, encouraging the model to stop early for easier tokens.
- **Gradient Detachment**: `model.detach_recur` can be used to detach the hidden states between recurrence steps, preventing BPTT across the full unrolled graph and saving memory.

### `NorMuon` Optimizer

`NorMuon` is a hybrid optimizer designed to handle different parameter types appropriately.

- **Manifold Optimization for Weights**: For weight matrices (2D tensors), it applies a momentum-based update on the Stiefel manifold. The gradient is projected onto its orthogonal component using the `polar_express` function (Newton-Schulz iteration). This enforces a structural constraint on the weights, which can improve stability and performance.
- **AdamW for Vectors**: For 1D parameters (biases, gains in normalization layers), it defaults to the well-established AdamW algorithm.
- **Distributed Version**: `DistNorMuon` efficiently shards gradients and updates across multiple GPUs, reducing communication overhead.

## Getting Started

### 1. Installation

The project uses `uv` for package management.

```bash
# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configuration

All experiment settings are managed in the `conf/` directory. You can create new YAML files or override settings directly from the command line.

**Example: Train a small standard GPT**
```yaml
# conf/model/gpt.yaml
name: gpt-small
d_model: 768
n_heads: 12
n_layers: 12
n_recur: 1 # n_recur=1 means it's a standard GPT
...
```

**Example: Train a recursive GPT**
```yaml
# conf/model/rgpt.yaml
name: rgpt-small
d_model: 768
n_heads: 12
n_layers: 6 # Fewer physical layers, but applied multiple times
n_recur: 4 # Recurrence depth
cal_halt_loss: true
detach_recur: false
...
```

### 3. Training

The training script `main.py` is the main entry point.

**Single-GPU Training:**
```bash
python main.py model=gpt # or model=rgpt
```

**Multi-GPU Training (DDP):**

The script uses environment variables (`WORLD_SIZE`, `LOCAL_RANK`) to set up DDP. Use a launcher like `torchrun`.

```bash
# Example: Training on a machine with 4 GPUs
torchrun --nproc_per_node=4 main.py \
	model=rgpt \
	training.optimizer.lr=1e-3 \
	dataset.train_batch_size=16
```

Outputs, including checkpoints and logs, will be saved to a timestamped directory under `outputs/`.

## Adding a New Model Architecture

While the framework is optimized for the provided `GPT` / `rGPT` architecture, it can be extended. Here’s how to add a new model type (e.g., a GRU-based sequence model).

### 1. Create the Model File

Create a new file in the `model/` directory, for example, `model/my_gru_model.py`. The class should accept a configuration object (`DictConfig`) for its parameters.

```python
# model/my_gru_model.py
import torch
import torch.nn as nn
from omegaconf import DictConfig

class MyGRUModel(nn.Module):
	def __init__(self, config: DictConfig):
		super().__init__()
		self.config = config
		self.embs = nn.Embedding(config.vocab_size, config.d_model)
		self.gru = nn.GRU(
			input_size=config.d_model,
			hidden_size=config.d_model,
			num_layers=config.n_layers,
			batch_first=True,
		)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

	def forward(self, input_ids, labels=None):
		x = self.embs(input_ids)
		x, _ = self.gru(x)
		logits = self.lm_head(x)

		# The trainer expects a tuple, even for simple models.
		# Return dummy values for unused loss components.
		return logits, torch.tensor(0.0), [], []
```

### 2. Create a Hydra Config

Create a corresponding configuration file in `conf/model/`. This file must include the `_target_` key, which points to the full path of your new class.

```yaml
# conf/model/my_gru.yaml

# Hydra's instantiation key
_target_: model.my_gru_model.MyGRUModel 

# --- Model Hyperparameters ---
name: "gru-medium"
vocab_size: 50257
d_model: 768
n_layers: 8
max_seq_len: 1024
pad_token_id: 50256
```

### 3. (Recommended) Refactor `main.py`

To make the model loading dynamic, modify the model instantiation line in `main.py` to use `hydra.utils.instantiate`. This allows Hydra to dynamically create an object from your config.

**From:**
```python
# --- Model ---
model = GPT(cfg.model)
```

**To:**
```python
# --- Model ---
# This will now use the `_target_` in your yaml to create the model object
model = hydra.utils.instantiate(cfg.model)
```
*Note: The `GPT` config files (`gpt.yaml`, `rgpt.yaml`) would also need to be updated to include `_target_: model.gpt.GPT`.*

### 4. Run Training

You can now train your new model by referencing its config file name.

```bash
# Train the new GRU model
python main.py model=my_gru
```
