from pathlib import Path
from omegaconf import DictConfig, OmegaConf


# Unified logger for TensorBoard and WandB
class UnifiedLogger:
	def __init__(self, cfg: DictConfig, hydra_output_dir: Path, rank: int):
		self.rank = rank
		self.writer = None
		if self.rank != 0:
			return

		self.logger_type = cfg.training.get("logger", "tensorboard")
		self.is_wandb = self.logger_type == "wandb"

		if self.is_wandb:
			try:
				import wandb
				from dotenv import load_dotenv

				wandb_cfg = cfg.get("wandb")
				if not wandb_cfg or not wandb_cfg.get("project"):
					print(
						"Warning: wandb logger enabled but 'wandb.project' not configured. Falling back to TensorBoard."
					)
					self.logger_type = "tensorboard"
					self.is_wandb = False
				else:
					load_dotenv()
					model_name = cfg.model.get("name", "model")
					task_name = cfg.task.get("name", "task")
					run_tag = cfg.get("run_tag")

					# Build a more descriptive name
					name_parts = [f"{task_name}-{model_name}"]
					if run_tag:
						name_parts.append(run_tag)

					# Add unique timestamp from hydra's output dir
					date_part = hydra_output_dir.parent.name
					time_part = hydra_output_dir.name
					name_parts.append(f"{date_part}_{time_part}")

					final_run_name = "-".join(name_parts)

					# Use the generated name unless a specific name is provided in the config
					run_name = cfg.wandb.get("name") or final_run_name

					wandb.init(
						project=wandb_cfg.project,
						name=run_name,
						config=OmegaConf.to_container(cfg, resolve=True),
						dir=str(hydra_output_dir),
					)
					self.writer = wandb
			except ImportError:
				print("wandb not installed, falling back to tensorboard")
				self.logger_type = "tensorboard"
				self.is_wandb = False

		if not self.is_wandb:
			from torch.utils.tensorboard import SummaryWriter

			self.writer = SummaryWriter(log_dir=str(hydra_output_dir))

	def log(self, data: dict, step: int):
		if self.rank == 0 and self.writer:
			if self.is_wandb:
				self.writer.log({**data, "epoch": step})
			else:
				for key, value in data.items():
					self.writer.add_scalar(key, value, step)

	def close(self):
		if self.rank == 0 and self.writer:
			if self.is_wandb:
				self.writer.finish()
			else:
				self.writer.close()
