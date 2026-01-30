import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import math
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from optimizer import Muon, DistMuon
from logger import UnifiedLogger


class Trainer:
	def __init__(
		self,
		cfg: DictConfig,
		model,
		train_loader,
		val_loader,
		test_loader,
		device,
		rank,
		world_size,
	):
		self.cfg = cfg
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.device = device
		self.rank = rank
		self.world_size = world_size
		self.is_ddp = self.world_size > 1
		self.is_main_process = self.rank == 0

		model = model.to(device)
		if self.cfg.training.get("compile", False):
			if self.is_main_process:
				print("Compiling the model...")
			model = torch.compile(model)

		if self.is_ddp:
			if self.device.type == "cuda":
				self.model = DDP(model, device_ids=[self.device.index])
			else:
				self.model = DDP(model)
		else:
			self.model = model

		self.dtype = getattr(torch, cfg.training.get("dtype", "bfloat16"))
		# self.use_amp = "cuda" in self.device.type
		self.use_amp = self.dtype == torch.float16
		self.scaler = GradScaler(enabled=(self.dtype == torch.float16))
		self.grad_accum_steps = self.cfg.training.get("grad_accum_steps", 1)
		self.early_stopping_patience = self.cfg.training.get("early_stopping_patience", 3)

		pad_id = self.cfg.model.get("pad_token_id", -100)
		self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
		self.eval_criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")

		# Setup optimizer groups
		muon_params, adamw_decay_params, adamw_no_decay_params = [], [], []
		tie_word_embeddings = self.cfg.model.get("tie_word_embeddings", True)
		for name, p in self.model.named_parameters():
			if not p.requires_grad:
				continue

			if not tie_word_embeddings and "lm_head" in name:
				adamw_decay_params.append(p)
				continue

			if "emb" in name:
				adamw_no_decay_params.append(p)
				continue

			if p.ndim >= 2:
				muon_params.append(p)
			else:
				if "norm" in name or "bias" in name:
					adamw_no_decay_params.append(p)
				else:
					adamw_decay_params.append(p)

		optimizer_cfg = self.cfg.training.optimizer
		muon_lr_multiplier = optimizer_cfg.get("muon_lr_multiplier", 100.0)
		param_groups = [
			{"params": muon_params, "use_muon": True},
			{"params": adamw_decay_params, "use_muon": False, "weight_decay": optimizer_cfg.weight_decay},
			{"params": adamw_no_decay_params, "use_muon": False, "weight_decay": 0.0},
		]

		optimizer_class = DistMuon if self.is_ddp else Muon
		self.optimizer = optimizer_class(
			param_groups,
			lr=optimizer_cfg.lr,
			weight_decay=optimizer_cfg.weight_decay,
			muon_lr_multiplier=muon_lr_multiplier,
		)

		# self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=optimizer_cfg.weight_decay)

		# Scheduler configuration logic
		scheduler_cfg = self.cfg.training.scheduler
		scheduler_name = scheduler_cfg.get("name", "linear")
		self.total_steps = self.cfg.training.total_steps
		warmup_steps = scheduler_cfg.get("warmup_steps", 1000)

		if scheduler_name == "linear":
			self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, self.total_steps)
		elif scheduler_name == "cosine":
			self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, self.total_steps)
		elif scheduler_name == "onecycle":
			adamw_max_lr = scheduler_cfg.get("max_lr", 0.01)
			max_lrs = [
				(adamw_max_lr * muon_lr_multiplier if g.get("use_muon") else adamw_max_lr)
				for g in self.optimizer.param_groups
			]
			self.scheduler = OneCycleLR(
				self.optimizer,
				max_lr=max_lrs,
				total_steps=self.total_steps,
				pct_start=scheduler_cfg.get("pct_start", 0.3),
				anneal_strategy="cos",
			)

		self.grad_clip = self.cfg.training.get("grad_clip", None)
		self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
		self.best_model_path = self.output_dir / f"best_{self.cfg.model.get('name', 'model')}.pth"

		self.best_metric = float("inf")
		self.best_epoch, self.epochs_no_improve, self.global_step = 0, 0, 0
		self.logger = UnifiedLogger(cfg, self.output_dir, self.rank)

	def compute_loss(self, outputs, labels):
		"""
		Computes the multi-objective loss for recursive reasoning models.
		"""
		if not isinstance(outputs, (list, tuple)) or len(outputs) == 1:
			return self.criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

		final_logits, accum_halt_loss, all_step_q, target_q_continues = outputs

		token_loss = self.criterion(final_logits.reshape(-1, final_logits.size(-1)), labels.reshape(-1))

		# Q-Head Losses (ACT logic)
		q_cont_loss = torch.tensor(0.0, device=self.device)

		if len(all_step_q) > 0:
			# target_q_continues is [n_recur, B, S]
			for i in range(len(all_step_q)):
				q_logits = all_step_q[i]
				q_continue_logits = q_logits[..., 1]

				# BCE with target from lookahead
				q_cont_loss += F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continues[i])

			q_cont_loss /= len(all_step_q)

		avg_halt_loss = accum_halt_loss / len(all_step_q) if len(all_step_q) > 0 else 0

		total_loss = token_loss + q_cont_loss + avg_halt_loss
		return total_loss, token_loss.item(), (q_cont_loss + avg_halt_loss).item()

	def train(self):
		metrics_dict = {"phase": "train", "best_loss": f"{self.best_metric:.4f}"}
		val_every_steps = self.cfg.training.get("val_every_n_steps", 1000)
		train_iter = iter(self.train_loader)

		with tqdm(total=self.total_steps, desc="Training Progress", disable=not self.is_main_process) as pbar:
			while self.global_step < self.total_steps:
				self.model.train()

				accum_total_loss = 0
				accum_token_loss = 0
				accum_q_loss = 0

				for _ in range(self.grad_accum_steps):
					try:
						batch = next(train_iter)
					except StopIteration:
						train_iter = iter(self.train_loader)
						batch = next(train_iter)

					input_ids = batch["src"].to(self.device, non_blocking=True)
					labels = batch["tgt"].to(self.device, non_blocking=True)

					with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
						# labels[:, 1:] because model predicts next token
						target_labels = labels[:, 1:].contiguous()
						outputs = self.model(input_ids[:, :-1], labels=target_labels)
						loss, t_loss, q_loss_val = self.compute_loss(outputs, labels[:, 1:])
						loss = loss / self.grad_accum_steps

					self.scaler.scale(loss).backward()

					accum_total_loss += loss.item()
					accum_token_loss += t_loss
					accum_q_loss += q_loss_val

				self.scaler.unscale_(self.optimizer)

				if self.is_main_process:
					# After loss.backward() but before optimizer step
					total_norm = 0
					for p in self.model.parameters():
						if p.grad is not None:
							param_norm = p.grad.data.norm(2)
							total_norm += param_norm.item() ** 2
					total_norm = total_norm**0.5
					print(f"Gradient norm: {total_norm:.4f}")

				# Optimizer step
				if self.grad_clip is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.optimizer.zero_grad(set_to_none=True)
				self.scheduler.step()

				self.global_step += 1
				pbar.update(1)

				if self.is_main_process:
					metrics_dict.update(
						{
							"loss": f"{accum_total_loss:.4f}",
							"q_loss": f"{accum_q_loss:.4f}",
							"lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
						}
					)
					pbar.set_postfix(metrics_dict)
					self.logger.log(
						{
							"Loss/train_total": accum_total_loss,
							"Loss/train_token": accum_token_loss / self.grad_accum_steps,
							"Loss/train_q": accum_q_loss,
							"lr": self.optimizer.param_groups[0]["lr"],
						},
						self.global_step,
					)

				# Validation
				if self.global_step > 0 and self.global_step % val_every_steps == 0:
					val_loss = self._validate_loss(self.val_loader)
					if self.is_main_process:
						metrics_dict["val_loss"] = f"{val_loss:.4f}"
						self.logger.log({"Loss/val_fineweb": val_loss}, self.global_step)
						if val_loss < self.best_metric:
							self.best_metric = val_loss
							self.best_epoch = self.global_step
							self.save_checkpoint(self.best_model_path, self.global_step)
							self.epochs_no_improve = 0
						else:
							self.epochs_no_improve += 1
						pbar.set_postfix(metrics_dict)

					if self.is_ddp:
						dist.barrier()
					if self.early_stopping_patience > 0 and self.epochs_no_improve >= self.early_stopping_patience:
						break

		if self.is_main_process:
			print(f"Training complete. Best Loss: {self.best_metric:.4f}")

	def _validate_loss(self, dataloader):
		self.model.eval()
		total_loss = torch.tensor(0.0, device=self.device)
		count = 0
		with torch.no_grad():
			for batch in dataloader:
				input_ids = batch["src"].to(self.device, non_blocking=True)
				labels = batch["tgt"].to(self.device, non_blocking=True)
				with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
					outputs = self.model(input_ids[:, :-1])
					# We only care about the final prediction loss for validation
					if isinstance(outputs, (list, tuple)):
						logits = outputs[0]
					else:
						logits = outputs
					loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
				total_loss += loss
				count += 1
		avg_loss = total_loss / count if count > 0 else total_loss
		if self.is_ddp:
			dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
		self.model.train()
		return avg_loss.item()

	def test(self):
		if self.is_main_process:
			print("Testing the model...")
		test_loss = self._validate_loss(self.test_loader)
		if self.is_main_process:
			self.logger.log({"Loss/test": test_loss}, self.global_step)
			print(f"Test Loss: {test_loss:.4f}")
		if self.is_ddp:
			dist.barrier()
		return test_loss

	def _get_model_for_saving(self):
		m = self.model.module if self.is_ddp else self.model
		return m._orig_mod if hasattr(m, "_orig_mod") else m

	def save_checkpoint(self, path, epoch):
		if not self.is_main_process:
			return
		torch.save(
			{
				"model_state_dict": self._get_model_for_saving().state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
				"best_metric": self.best_metric,
				"global_step": self.global_step,
			},
			path,
		)
