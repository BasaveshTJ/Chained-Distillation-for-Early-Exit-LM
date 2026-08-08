import torch

from datasets import DatasetDict, load_dataset

from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from loss_fn import hierarchical_early_exit_kd_loss
from SmolLM2EarlyExitForCausalLM import SmolLM2EarlyExitForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR = "./smollm2-135m-finetuned"
SUBSET = "everyday-conversations"

# ============================================================
# Early-exit configuration
# ============================================================

EXIT_LAYERS = [10, 20, 30]
TEMPERATURE = 2.0
LAMBDA_KD = [
    1.0,   # EXIT 10 <- EXIT 20
    1.0,   # EXIT 20 <- EXIT 30
]
LAMBDA_CE = 1.0

# ============================================================
# KD memory configuration
# ============================================================

# Vocabulary = 49152.
#
# The loss processes the vocabulary in chunks instead of
# creating full softmax/log-softmax tensors.
#
# Smaller values use less temporary GPU memory.
#
VOCAB_CHUNK_SIZE = 1024

# ============================================================
# Training configuration
# ============================================================

# Original:
#
# per_device_train_batch_size = 32
# gradient_accumulation_steps = 1
#
# Effective batch = 32
#
# New:
#
# per_device_train_batch_size = 8
# gradient_accumulation_steps = 4
#
# Effective batch = 32
#

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

# ============================================================
# Tokenizer
# ============================================================

print(f"🚀 Loading tokenizer and model: {MODEL_ID}")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ============================================================
# Model
# ============================================================

print(f"Loading pretrained model: {MODEL_ID}")
model = SmolLM2EarlyExitForCausalLM.from_pretrained( MODEL_ID, dtype=torch.float32, attn_implementation="sdpa",)

print("Model dtype:", next(model.parameters()).dtype)
print("CUDA device:", torch.cuda.get_device_name())
print("BF16 supported:", torch.cuda.is_bf16_supported())


# ============================================================
# Architecture verification
# ============================================================
model.verify_architecture()

# ============================================================
# Dataset
# ============================================================

print(f"📦 Loading dataset split: {DATASET_ID}")
train_dataset = load_dataset(DATASET_ID, split="train")
test_dataset = load_dataset(DATASET_ID, split="test")

# ============================================================
# Select SmolLM rewrite subset
# ============================================================

train_dataset = train_dataset.filter(lambda example: example["source"] == SUBSET)
test_dataset = test_dataset.filter(lambda example: example["source"] == SUBSET)

print(f"Training examples: {len(train_dataset)}")
print(f"Test examples: {len(test_dataset)}")

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# ============================================================
# Chat formatting
# ============================================================

def format_prompts(batch):
    texts = []

    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

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
    return hierarchical_early_exit_kd_loss(
        exit_logits=outputs.exit_logits,
        labels=labels,
        temperature=TEMPERATURE,
        lambda_kd=LAMBDA_KD,
        lambda_ce=LAMBDA_CE,
        vocab_chunk_size=VOCAB_CHUNK_SIZE,
    )

# ============================================================
# Training arguments
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    optim="adamw_torch_fused",
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={ "use_reentrant": False },
    report_to="none",
    seed=42,
    max_length=2048,
    dataset_text_field="text",
    ddp_find_unused_parameters=False,
)


print("bf16:", training_args.bf16)
print("fp16:", training_args.fp16)
print("torch dtype:", next(model.parameters()).dtype)
print("\nExit layers:", EXIT_LAYERS)
print("Temperature:", TEMPERATURE)
print("Lambda KD:", LAMBDA_KD)
print("Lambda CE:", LAMBDA_CE)
print("Vocabulary chunk size:", VOCAB_CHUNK_SIZE)

print("=" * 70)


# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    processing_class=tokenizer,
    compute_loss_func=compute_loss_func,
)


# ============================================================
# Train
# ============================================================

print("🏋️ Starting fine-tuning loop...")

trainer.train()


# ============================================================
# Save
# ============================================================

print(f"💾 Saving fine-tuned weights to {OUTPUT_DIR}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning completed successfully!")