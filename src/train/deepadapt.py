import torch
import torch.nn.functional as F
import importlib.util
import os
import sys

from datasets import DatasetDict, load_dataset

from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(SRC_DIR, "model")
if MODEL_DIR not in sys.path:
	sys.path.insert(0, MODEL_DIR)

MODEL_FILE = os.path.join(MODEL_DIR, "SmolLM2DeepAdaptForCausalLM.py")
spec = importlib.util.spec_from_file_location("deepadapt_model", MODEL_FILE)
if spec is None or spec.loader is None:
	raise ImportError(f"Could not load model file: {MODEL_FILE}")
deepadapt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepadapt_module)
SmolLM2DeepAdaptForCausalLM = deepadapt_module.SmolLM2DeepAdaptForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR = "./smollm2-135m-deepadapt"
SUBSET = "everyday-conversations"

# ============================================================
# DeepAdapt configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.5

# ============================================================
# Training configuration
# ============================================================

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4


# ============================================================
# Tokenizer
# ============================================================

print(f"Loading tokenizer and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# Model
# ============================================================

print(f"Loading pretrained model: {MODEL_ID}")
model = SmolLM2DeepAdaptForCausalLM.from_pretrained(
	MODEL_ID,
	dtype=torch.float32,
	attn_implementation="sdpa",
)
model.set_confidence_threshold(CONFIDENCE_THRESHOLD)
model.verify_trainable_parameters()

print("Model dtype:", next(model.parameters()).dtype)
if torch.cuda.is_available():
	print("CUDA device:", torch.cuda.get_device_name())
	print("BF16 supported:", torch.cuda.is_bf16_supported())


# ============================================================
# Dataset
# ============================================================

print(f"Loading dataset split: {DATASET_ID}")
train_dataset = load_dataset(DATASET_ID, split="train")
test_dataset = load_dataset(DATASET_ID, split="test")


# ============================================================
# Select subset
# ============================================================

train_dataset = train_dataset.filter(lambda example: example["source"] == SUBSET)
test_dataset = test_dataset.filter(lambda example: example["source"] == SUBSET)

print(f"Training examples: {len(train_dataset)}")
print(f"Test examples: {len(test_dataset)}")

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
raw_columns = dataset["train"].column_names


# ============================================================
# Chat formatting
# ============================================================

def _fallback_chat_format(messages):
	lines = []
	for message in messages:
		role = message["role"].capitalize()
		content = message["content"]
		lines.append(f"{role}: {content}")
	lines.append("Assistant:")
	return "\n".join(lines)


def format_prompts(batch):
	texts = []

	for messages in batch["messages"]:
		if tokenizer.chat_template is not None:
			text = tokenizer.apply_chat_template(messages, tokenize=False)
		else:
			text = _fallback_chat_format(messages)
		texts.append(text)

	return {"text": texts}


dataset = dataset.map(
	format_prompts,
	batched=True,
	remove_columns=raw_columns,
)


# ============================================================
# Train / validation split
# ============================================================

train_val = dataset["train"].train_test_split(
	test_size=0.05,
	seed=42,
)

dataset_dict = DatasetDict({
	"train": train_val["train"],
	"validation": train_val["test"],
	"test": dataset["test"],
})

print("\nDataset:")
print(f"Train: {len(dataset_dict['train'])}")
print(f"Validation: {len(dataset_dict['validation'])}")
print(f"Test: {len(dataset_dict['test'])}")


# ============================================================
# Custom loss
# ============================================================

def compute_loss_func(
	outputs,
	labels,
	num_items_in_batch=None,
):
	shift_logits = outputs.logits[..., :-1, :]
	shift_labels = labels[..., 1:]

	valid_mask = shift_labels != -100
	low_conf_mask = outputs.low_conf_mask[:, :-1]
	train_mask = valid_mask & low_conf_mask

	if train_mask.sum() == 0:
		# Keep graph connected when no token is selected.
		return shift_logits.sum() * 0.0

	per_token_loss = F.cross_entropy(
		shift_logits.reshape(-1, shift_logits.size(-1)),
		shift_labels.reshape(-1),
		ignore_index=-100,
		reduction="none",
	).view_as(shift_labels)

	masked_loss = per_token_loss * train_mask.to(dtype=per_token_loss.dtype)
	return masked_loss.sum() / train_mask.sum().to(dtype=per_token_loss.dtype)


# ============================================================
# Training arguments
# ============================================================

training_args = SFTConfig(
	output_dir=OUTPUT_DIR,
	per_device_train_batch_size=TRAIN_BATCH_SIZE,
	per_device_eval_batch_size=EVAL_BATCH_SIZE,
	gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
	learning_rate=2e-4,
	lr_scheduler_type="cosine",
	warmup_steps=100,
	optim="adamw_torch_fused",
	logging_steps=100,
	eval_strategy="epoch",
	save_strategy="epoch",
	num_train_epochs=3,
	bf16=False,
	fp16=True,
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": False},
	report_to="none",
	seed=42,
	max_length=2048,
	dataset_text_field="text",
	ddp_find_unused_parameters=False,
)

print("bf16:", training_args.bf16)
print("fp16:", training_args.fp16)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("=" * 70)


# ============================================================
# Train
# ============================================================

trainer = SFTTrainer(
	model=model,
	args=training_args,
	train_dataset=dataset_dict["train"],
	eval_dataset=dataset_dict["validation"],
	processing_class=tokenizer,
	compute_loss_func=compute_loss_func,
)


print("Starting fine-tuning loop...")
trainer.train()


print(f"Saving fine-tuned weights to {OUTPUT_DIR}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed successfully!")
