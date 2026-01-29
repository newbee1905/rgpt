import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import itertools
from functools import partial


class FineWebDataset(IterableDataset):
	def __init__(
		self, tokenizer: PreTrainedTokenizer, max_seq_len: int, split: str = "train", val_samples: int = 100_000
	):
		"""
		Args:
			val_samples: Number of documents to hold out for validation.
						 100k docs ~= 100M tokens (assuming ~1k tokens/doc).
		"""
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len

		# 1. Always load the full 'train' split since it's the only one available
		hf_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

		if split == "validation":
			self.dataset = hf_dataset.take(val_samples)
		else:
			self.dataset = hf_dataset.skip(val_samples)

	def __iter__(self):
		buffer = []
		for sample in self.dataset:
			text = sample["text"]

			# Tokenize the text by chunks to avoid very long token sequences.
			for i in range(0, len(text), 1024 * 100):
				chunk_text = text[i : i + 1024 * 100]
				tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
				buffer.extend(tokens)
				while len(buffer) >= self.max_seq_len:
					chunk = buffer[: self.max_seq_len]
					buffer = buffer[self.max_seq_len :]
					yield torch.tensor(chunk)


def fineweb_collate_fn(batch):
	"""
	Collate function for FineWebDataset.
	'src' and 'tgt' are the same for autoregressive training.
	"""
	stacked_batch = torch.stack(batch)
	return {"src": stacked_batch, "tgt": stacked_batch}


class HellaswagDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, split="validation"):
		self.tokenizer = tokenizer
		self.dataset = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		story = self.dataset[idx]
		context = story["ctx"]
		endings = story["endings"]
		label = int(story["label"])

		return {"context": context, "endings": endings, "label": label}


def hellaswag_collate_fn(batch, tokenizer: PreTrainedTokenizer, max_seq_len: int):
	"""
	Collate function for HellaswagDataset.
	Creates multiple choice inputs for each example in the batch.
	"""
	all_inputs = []
	all_labels = []

	pad_token_id = tokenizer.pad_token_id
	if pad_token_id is None:
		if tokenizer.eos_token_id is not None:
			pad_token_id = tokenizer.eos_token_id
		else:
			raise ValueError("Tokenizer must have a pad_token_id or eos_token_id defined.")

	bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
	eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

	bos_seq = [bos_token_id] if bos_token_id is not None else []
	eos_seq = [eos_token_id] if eos_token_id is not None else []

	for item in batch:
		context = item["context"]
		label = item["label"]

		# Note: add_special_tokens=False because we manually add BOS/EOS later
		context_tokens = tokenizer.encode(context, add_special_tokens=False)

		item_choices = []
		for i, ending in enumerate(item["endings"]):
			ending_tokens = tokenizer.encode(ending, add_special_tokens=False)

			# Construct sequence
			input_tokens = bos_seq + context_tokens + ending_tokens + eos_seq

			# Truncate from the left if necessary
			if len(input_tokens) > max_seq_len:
				input_tokens = input_tokens[-max_seq_len:]

			seq_len = len(input_tokens)

			input_ids = torch.full((max_seq_len,), pad_token_id, dtype=torch.long)
			input_ids[:seq_len] = torch.tensor(input_tokens)

			labels = input_ids.clone()

			# Mask out context and padding
			# Calculate where the ending starts (accounting for EOS length)
			ending_len = len(ending_tokens) + len(eos_seq)
			context_len = seq_len - ending_len

			# Mask context
			if context_len > 0:
				labels[:context_len] = -100

			# Mask padding (everything after seq_len)
			labels[seq_len:] = -100

			item_choices.append((input_ids, labels))

		all_inputs.append(item_choices)
		all_labels.append(label)

	flat_input_ids = []
	flat_labels = []
	example_indices = []

	for i, choices in enumerate(all_inputs):
		for input_ids, labels in choices:
			flat_input_ids.append(input_ids)
			flat_labels.append(labels)
			example_indices.append(i)

	return {
		"input_ids": torch.stack(flat_input_ids).contiguous(),
		"labels": torch.stack(flat_labels).contiguous(),
		"example_indices": torch.tensor(example_indices, dtype=torch.long),
		"correct_labels": torch.tensor(all_labels, dtype=torch.long),
	}


def get_collate_fn(tokenizer, max_seq_len):
	return partial(hellaswag_collate_fn, tokenizer=tokenizer, max_seq_len=max_seq_len)


if __name__ == "__main__":
	from transformers import AutoTokenizer

	# --- Test FineWebDataset ---
	print("--- Testing FineWebDataset ---")
	tokenizer = AutoTokenizer.from_pretrained("gpt2")

	fw_dataset = FineWebDataset(tokenizer=tokenizer, max_seq_len=128)

	print("Taking 2 samples from FineWebDataset...")
	for i, sample in enumerate(fw_dataset):
		print(f"Sample {i + 1} shape: {sample.shape}")
		if i >= 1:
			break
	print("-" * 20)

	# --- Test HellaswagDataset ---
	print("\n--- Testing HellaswagDataset ---")
	hs_dataset = HellaswagDataset(tokenizer=tokenizer)
	print(f"Hellaswag dataset size: {len(hs_dataset)}")

	print("\nTesting hellaswag_collate_fn...")
	collate_fn = get_collate_fn(tokenizer, max_seq_len=256)

	batch = [hs_dataset[0], hs_dataset[1]]
	collated_batch = collate_fn(batch)

	print("Collated batch keys:", collated_batch.keys())
	print("Shape of 'input_ids':", collated_batch["input_ids"].shape)

	# Check what value was used for padding
	# We expect 50256 (EOS ID for GPT-2) since PAD is None
	pad_value = collated_batch["input_ids"][0, -1].item()
	print(f"Value used for padding: {pad_value} (GPT-2 EOS is {tokenizer.eos_token_id})")

	if pad_value == tokenizer.eos_token_id:
		print("Success: EOS token was used as fallback for padding.")
	else:
		print(f"Note: Padding used {pad_value}")

	print("\nDataset tests seem to be working.")
