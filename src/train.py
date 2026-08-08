import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR = "./smollm2-135m-finetuned"

print(f"🚀 Loading tokenizer and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Ensure correct padding token configuration for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# Load base model in bfloat16 for stability
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    attn_implementation="sdpa",
)

print("Model dtype:", next(model.parameters()).dtype)
print("CUDA device:", torch.cuda.get_device_name())
print("BF16 supported:", torch.cuda.is_bf16_supported())

# 2. Load and Prepare the Dataset
print(f"📦 Loading dataset split: {DATASET_ID}")

train_dataset = load_dataset(DATASET_ID, split="train")
test_dataset = load_dataset(DATASET_ID, split="test")
_subset = "smollm-rewrite-30k"  # 26657
# select only teh subset of the dataset for training and testing
train_dataset = train_dataset.filter(lambda example: example["source"] == _subset)
test_dataset = test_dataset.filter(lambda example: example["source"] == _subset)

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# 3. Format dataset using SmolLM2's internal Chat Template
def format_prompts(batch):
    texts = []
    for messages in batch["messages"]:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# Split for evaluation validation
train_val = dataset["train"].train_test_split( test_size=0.05,seed=42)
dataset_dict = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": dataset["test"],
})

#  Training Hyperparameters
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    optim="adamw_torch_fused",
    seed=42,
    max_length=2048,
    dataset_text_field="text",
    ddp_find_unused_parameters=False,
)

print("bf16:", training_args.bf16)
print("fp16:", training_args.fp16)
print("torch dtype:", next(model.parameters()).dtype)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    processing_class=tokenizer,
)

# 7. Start Training
print("🏋️ Starting fine-tuning loop...")

trainer.train()

# 8. Save fine-tuned checkpoint & tokenizer
print(f"💾 Saving fine-tuned weights to {OUTPUT_DIR}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning completed successfully!")