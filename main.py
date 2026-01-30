import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import os

from trainer import Trainer
from dataset import FineWebDataset, HellaswagDataset, fineweb_collate_fn, get_collate_fn

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def setup_ddp():
	"""Initializes the distributed data parallel environment."""
	if "WORLD_SIZE" in os.environ:
		world_size = int(os.environ["WORLD_SIZE"])
		rank = int(os.environ["LOCAL_RANK"])
		dist.init_process_group(backend="nccl")
		torch.cuda.set_device(rank)
		is_ddp = True
	else:
		world_size = 1
		rank = 0
		is_ddp = False
	return world_size, rank, is_ddp


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
	"""Main training entry point."""
	torch.backends.cudnn.benchmark = True
	world_size, rank, is_ddp = setup_ddp()
	if rank == 0:
		print(OmegaConf.to_yaml(cfg))

	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

	# --- Tokenizer ---
	tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_path)

	# Update pad_token_id in config if not set, important for collate functions
	if "pad_token_id" not in cfg.model or cfg.model.pad_token_id is None:
		cfg.model.pad_token_id = (
			tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
		)

	# --- Datasets ---
	train_dataset = FineWebDataset(tokenizer, cfg.model.max_seq_len, split="train")
	val_dataset = FineWebDataset(tokenizer, cfg.model.max_seq_len, split="validation")
	test_dataset = HellaswagDataset(tokenizer, split="validation")

	# --- Dataloaders ---

	train_loader = DataLoader(
		train_dataset,
		batch_size=cfg.dataset.train_batch_size,
		collate_fn=fineweb_collate_fn,
		num_workers=cfg.dataset.num_workers,
		pin_memory=True,
		shuffle=False,
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=cfg.dataset.val_batch_size,
		collate_fn=fineweb_collate_fn,
		num_workers=cfg.dataset.num_workers,
		pin_memory=True,
		shuffle=False,
	)

	# Hellaswag is a Map-style dataset, so we can use DistributedSampler if needed for testing,
	# though usually for testing on rank 0 or with gather is fine.
	# For simplicity in DDP, we can use sampler to split work.
	test_sampler = (
		DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=cfg.dataset.val_batch_size,
		sampler=test_sampler,
		shuffle=False,
		collate_fn=get_collate_fn(tokenizer, cfg.model.max_seq_len),
		num_workers=cfg.dataset.num_workers,
	)

	# --- Model ---
	model = hydra.utils.instantiate(cfg.model)

	# --- Calculate total training steps based on scaling laws ---
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if rank == 0:
		print(f"Model has {num_params / 1e6:.2f}M trainable parameters.")

	# For recurrent models, calculate an "effective" number of parameters to align training token counts
	# with non-recurrent models of similar computational cost (total layers * recurrence).
	n_recur = cfg.model.get("n_recur", 1)
	if n_recur > 1:
		layer_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "layers." in n)
		other_params = num_params - layer_params
		effective_params = other_params + (layer_params * n_recur)
		if rank == 0:
			print(f"Recurrent model detected (n_recur={n_recur}). Using effective parameter count for token scaling.")
			print(f"  - Effective params: {effective_params / 1e6:.2f}M")
	else:
		effective_params = num_params

	total_training_tokens = effective_params * 20

	# Considering DDP, batch size is per-device.
	tokens_per_step = cfg.dataset.train_batch_size * cfg.model.max_seq_len * cfg.training.grad_accum_steps * world_size
	total_steps = int(total_training_tokens // tokens_per_step)

	if rank == 0:
		print(f"Following scaling laws (20 tokens/param):")
		print(f"  - Total training tokens: {total_training_tokens / 1e9:.2f}B")
		print(f"  - Total training steps: {total_steps}")

	# Pass total_steps to trainer config
	OmegaConf.set_struct(cfg, False)
	cfg.training.total_steps = total_steps
	OmegaConf.set_struct(cfg, True)

	# --- Trainer and Training ---
	trainer = Trainer(
		cfg=cfg,
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		device=device,
		rank=rank,
		world_size=world_size,
	)

	# Run Training Loop (Validates on FineWeb)
	trainer.train()

	# Run Final Test (Evaluates on Hellaswag)
	trainer.test()


if __name__ == "__main__":
	main()
