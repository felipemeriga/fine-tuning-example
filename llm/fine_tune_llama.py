"""
Fine-tune LLaMA 3.2 1B for chat using QLoRA (4-bit quantization + LoRA adapters).

== How this differs from the sentiment fine-tuning ==
- Sentiment model: classification (sentence -> positive/negative label)
- This model: generative chat (prompt -> multi-token response, next-token prediction)

== QLoRA — how to fine-tune a large model on a small GPU ==
Full fine-tuning updates ALL weights (like we did with DistilBERT's 66M params).
That's impossible for a 1B param model on 8GB VRAM — the weights alone would be ~4GB in fp16.
QLoRA solves this in two steps:

1. Quantization (the "Q"):
   Load the model in 4-bit precision (~0.5 bytes per param instead of 2).
   1B params * 0.5 bytes = ~500MB instead of ~2GB. The model is FROZEN — these weights
   don't change during training.

2. LoRA adapters (Low-Rank Adaptation):
   Instead of updating the full weight matrices, inject small trainable adapter matrices.
   A weight matrix W (4096x4096 = 16M params) is approximated by two small matrices:
     W_adapted = W_frozen + A x B    where A is (4096x16) and B is (16x4096)
   That's only 131K params instead of 16M — a 99% reduction.
   Only A and B are trained. The original W stays frozen in 4-bit.

Result: ~1% of parameters are trainable, VRAM usage drops dramatically,
and the quality is surprisingly close to full fine-tuning.

== The training data format ==
For chat fine-tuning, we use conversation pairs formatted with the model's chat template:
  <|begin_of_text|><|start_header_id|>user<|end_header_id|>
  What is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  Python is a programming language...<|eot_id|>

The model learns to predict the ASSISTANT tokens given the USER tokens as context.
This is still next-token prediction — the same mechanism as pre-training, just on
curated prompt/response pairs so the model learns to be helpful.

Run with:
    uv run python llm/fine_tune_llama.py

Requires:
    - NVIDIA GPU with at least 8GB VRAM
    - Uses TinyLlama 1.1B (open access, no approval needed)
    - Swap to meta-llama/Llama-3.2-1B-Instruct once you have Meta's approval
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# --- Configuration ---
# TinyLlama: same architecture as LLaMA, 1.1B params, open access (no approval needed).
# Swap to "meta-llama/Llama-3.2-1B-Instruct" once you have Meta's approval.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./llm/llama-chat-finetuned"
# Using a small, high-quality instruction dataset
DATASET_NAME = "mlabonne/guanaco-llama2-1k"
MAX_LENGTH = 512
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch size = 2 * 8 = 16
LEARNING_RATE = 2e-4  # higher than sentiment fine-tuning because only LoRA adapters are trained

# --- LoRA config ---
# r (rank): size of the adapter matrices. Higher = more capacity but more VRAM.
# lora_alpha: scaling factor. Usually set to 2x the rank.
# target_modules: which layers get adapters. For LLaMA, we target the attention projections.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for QLoRA training.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- Quantization config ---
    # Load the model in 4-bit precision using NF4 (Normal Float 4), a data type
    # specifically designed for normally distributed neural network weights.
    # double_quant: quantize the quantization constants too (saves ~0.4 bits/param).
    # compute_dtype: use fp16 for the actual computation (4-bit is just for storage).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load tokenizer ---
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # pad_token is needed for batching. LLaMA doesn't have one by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load model in 4-bit ---
    # The model weights are quantized as they're loaded — no full-precision copy in RAM.
    # ~1B params at 4-bit = ~500MB VRAM (vs ~2GB at fp16 or ~4GB at fp32).
    print(f"Loading {MODEL_NAME} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Prepare for training: enable gradient checkpointing (trades compute for VRAM)
    # and handle quantized layer normalization.
    model = prepare_model_for_kbit_training(model)

    # --- Add LoRA adapters ---
    # This injects small trainable matrices into the specified layers.
    # Only these adapters will be trained — the base model stays frozen.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        # Target the attention projection layers in each transformer block
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load dataset ---
    # This dataset contains instruction/response pairs formatted for chat.
    print(f"\nLoading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"  Training examples: {len(dataset):,}")

    # --- Training args ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        # Gradient accumulation: simulate a larger batch size without using more VRAM.
        # Process 2 examples at a time, accumulate gradients for 8 steps, then update.
        # Effective batch size = 2 * 8 = 16.
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # Gradient checkpointing: recompute activations during backward pass instead
        # of storing them. Uses less VRAM but slightly slower training.
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # 8-bit optimizer to save VRAM
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        report_to="none",
    )

    # --- Train ---
    # SFTTrainer (Supervised Fine-Tuning Trainer) from TRL handles chat formatting.
    # It tokenizes the dataset, applies the chat template, and masks the user tokens
    # so the loss is only computed on the assistant's response (the model shouldn't
    # be penalized for not predicting the user's prompt).
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting QLoRA fine-tuning...\n")
    trainer.train()

    # --- Save ---
    # Only the LoRA adapter weights are saved (~10-50MB), not the full model.
    # To use it later, you load the base model + merge the adapters on top.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter weights saved to {OUTPUT_DIR}/")

    # --- Quick inference demo ---
    print("\n--- Inference Demo ---")
    # Reload base model in 4-bit and merge the trained adapters
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    prompts = [
        "What is machine learning in simple terms?",
        "Write a haiku about programming.",
    ]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"\nUser: {prompt}")
        print(f"Assistant: {response.strip()}")


if __name__ == "__main__":
    main()
