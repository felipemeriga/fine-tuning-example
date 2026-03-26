"""
Fine-tune DistilBERT on SST-2 (Stanford Sentiment Treebank) for binary sentiment analysis.
Uses Hugging Face transformers + datasets, with GPU acceleration and fp16 mixed precision.

== What is fine-tuning? ==
A pre-trained model already learned language structure from billions of words.
Fine-tuning adapts that knowledge to a specific task (here: sentiment classification)
using a smaller labeled dataset. Instead of training from scratch, we just teach the "last mile."

== How this differs from LLM training ==
This is a CLASSIFICATION model — input is a sentence, output is a label (positive/negative).
An LLM is a GENERATIVE model — it predicts the next token, and the training text IS the label:

  Training text: "The cat sat on the mat"
  Position:       The   cat   sat   on    the
  Label:          cat   sat   on    the   mat

Each token is both part of the input (context for future tokens) and a label (target for the
previous position). A causal mask ensures each position only sees tokens to its left — no
cheating by peeking ahead. Loss is computed at all positions simultaneously in one forward pass.

Self-attention (context) is used in BOTH training and inference — it's the same mechanism.
During training, the causal mask gives each position its context:
  Position "mat" attends to: [The, cat, sat, on, the]  <- its context
  Position "sat" attends to: [The, cat]                 <- its context
The model LEARNS how to use context — the attention weights are trained to figure out which
previous tokens matter most for predicting the next one.

== LLM Training vs Inference ==
Training:
  - Full sequence goes in at once (all positions processed in parallel)
  - Causal mask prevents each position from seeing future tokens
  - Context exists for all positions simultaneously
  - Loss is computed at every position, backpropagation updates weights
  - The model needs massive amounts of text because after any token (e.g., "the"),
    many different words might follow — it must see enough examples to learn good
    probability distributions for every possible context

Inference (e.g., ChatGPT answering a question):
  - Token by token — each step requires a forward pass because future tokens don't exist yet
  - The user's prompt is just tokens on the left (context), and the model completes it
    by predicting the most probable next token, one at a time
  - It "answers" questions because it was fine-tuned on chat data (millions of prompt/response
    pairs), so given "User: [question]\nAssistant:", it learned that a helpful answer
    is the most probable continuation — it's still just next-token prediction
  - A KV cache stores previously computed attention keys/values to avoid reprocessing
    the entire sequence at each step
  - No loss, no gradients, no weight updates — the model is frozen

This classification model uses BIDIRECTIONAL attention — every token sees every other token,
because for classification you want full context, not left-to-right generation.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

# --- Hyperparameters ---
# DistilBERT: distilled version of BERT, 40% smaller, 60% faster, retains ~97% performance.
# ~66M parameters, fits in 8GB VRAM. "uncased" lowercases everything.
MODEL_NAME = "distilbert-base-usncased"
OUTPUT_DIR = "./sentiment-model"
# 3 epochs: one epoch = one full pass through all training examples. BERT paper recommends 2-4
# for fine-tuning. Too few = underfitting, too many = overfitting (memorizing instead of learning).
NUM_EPOCHS = 3
# Batch size: how many examples the model sees before updating weights. Larger = more stable
# gradients but more VRAM. 32 is a sweet spot for 8GB.
BATCH_SIZE = 32
# Learning rate: deliberately tiny (0.00002). The model already has good language understanding;
# a large learning rate would destroy pre-trained knowledge (catastrophic forgetting).
LEARNING_RATE = 2e-5
# Max sequence length in tokens (not words). Sentences shorter than this get padded,
# longer get truncated.
MAX_LENGTH = 128


def main():
    # --- Device detection ---
    # GPUs run thousands of parallel operations (matrix multiplications), which is exactly
    # what neural networks need. Training on GPU is typically 10-50x faster than CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load dataset ---
    # SST-2 (Stanford Sentiment Treebank) is part of the GLUE benchmark.
    # Binary sentiment: each movie review sentence is labeled 0 (negative) or 1 (positive).
    # This is SUPERVISED learning — every example comes paired with its expected result (label).
    # Without labels, the model would have no way to know if it was right or wrong.
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")

    # --- Tokenize ---
    # Load the tokenizer that matches our model — same vocabulary and rules used during pre-training.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    '''
      Tokenization: text -> subword IDs using WordPiece (~30,522 vocabulary).
      Common words map to one token ("movie" -> [3185]).
      Rare words are split into subwords ("unhappiness" -> ["un", "##happi", "##ness"]).
      The ## prefix means "continuation of the previous word, not a standalone token."
      Without ##, the model couldn't tell "unhappiness" (one word split) from "un happi ness" (three words).

      A sequence is the full list of token IDs for one sentence:
      "This movie was great" -> [CLS, This, movie, was, great, SEP, PAD, PAD, ..., PAD]
                                 |_____________ 128 tokens total ___________________|

      - padding="max_length": pad short sequences to 128 tokens (GPUs need fixed-size matrices)
      - truncation=True: cut sentences longer than 128 tokens
      - An attention mask (1s/0s) is also generated so the model ignores padding

      A batch is 32 sequences stacked into a (32, 128) matrix.
      Labels are already integers (0/1), they pass through untouched.
    '''
    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing...")
    # Apply tokenize() to every example. batched=True processes them in chunks for speed.
    # remove_columns drops "sentence" (raw text) and "idx" (index) — the model only needs
    # input_ids, attention_mask, and the label. The raw text is no longer useful after tokenization.
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    # Rename "label" to "labels" — the Trainer expects this exact column name
    # to automatically compute the loss (it passes it as the `labels` argument to the model).
    tokenized = tokenized.rename_column("label", "labels")
    # Convert to PyTorch tensors so they can be fed directly to the model.
    tokenized.set_format("torch")

    # --- Load model ---
    # from_pretrained loads the pre-trained weights — language understanding learned from
    # Wikipedia + BookCorpus. The model body (6 transformer layers) is pre-trained.
    #
    # AutoModelForSequenceClassification adds a CLASSIFICATION HEAD on top:
    # two linear layers that take the [CLS] token's final 768-dim vector and output 2 logits
    # (one per class). This head is RANDOMLY INITIALIZED — it's the part that needs the most learning.
    #
    # == What happens inside the model (forward pass) ==
    # 1. Embedding layer: token IDs -> 768-dim vectors (learned lookup table, trainable).
    #    Also adds position embeddings so the model knows word order.
    # 2. 6 Transformer layers: each applies self-attention + feed-forward network.
    #    Residual connections prevent vanishing gradients.
    # 3. Classification head: [CLS] token's 768-dim output -> 2 logits (negative, positive).
    #
    # Every layer is built from the same primitive: output = activation(W x input + bias)
    # The activation (GELU) adds non-linearity — without it, stacking layers would collapse
    # into a single linear transformation.
    #
    # == Softmax is used TWICE, for different purposes ==
    #
    # 1) INSIDE self-attention (every layer, internal):
    #    Each token computes Q (query), K (key), V (value) from W x input + bias.
    #    scores = softmax(Q x K^T)  <- attention scores: how much each token attends to others
    #    output = scores x V        <- weighted mix of value vectors
    #    These are NOT predictions. They're internal weights for mixing token representations.
    #    The W parameters are fixed at inference, but the scores are computed fresh for each input
    #    (same recipe, different ingredients = different scores).
    #
    # 2) AT THE END (output, prediction):
    #    logits -> softmax -> probabilities for each class
    #    Classification: [-1.8, 0.3] -> [0.11, 0.89]  (2 values: negative, positive)
    #    LLM:            [2.1, -0.3, 5.2, ...] -> [0.01, 0.001, 0.24, ...]  (50,000+ values,
    #                    one per vocabulary token — probability distribution over ALL possible
    #                    next tokens)
    #
    # Same formula both times (e^x / sum(e^x)), completely different purpose:
    #   Input -> [Attention softmax (x6 layers)] -> Logits -> Output softmax -> Prediction
    #             (internal mixing)                            (actual prediction)
    #
    # During TRAINING:
    #   - Both softmaxes run. Attention scores mix representations, output softmax produces
    #     probabilities, loss is computed against the true label, backpropagation updates
    #     all weights (including the W_query, W_key, W_value matrices in attention).
    #   - For LLMs: output softmax produces 50,000+ probabilities at EVERY position simultaneously.
    #     Loss is computed at each position (predicted token vs actual next token).
    #
    # During INFERENCE:
    #   - Both softmaxes still run, same mechanism. But no loss, no gradients, no weight updates.
    #   - For classification: output softmax gives 2 probabilities, pick the highest.
    #   - For LLMs: output softmax gives 50,000+ probabilities for the next token.
    #     A sampling strategy picks one (greedy = always top, top-k = random from top k,
    #     temperature = controls randomness). That's why LLMs can give different answers
    #     to the same prompt — they sample from a distribution.
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # --- Metrics ---
    # The model outputs LOGITS — raw unnormalized scores (e.g., [-1.8, 0.3]).
    # argmax picks the class with the highest score as the prediction.
    # Accuracy: % of correct predictions.
    # F1: harmonic mean of precision and recall — more robust with imbalanced classes.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LEARNING_RATE,
        # Weight decay: regularization that penalizes large weights, nudging them toward zero.
        # Prevents the model from relying too heavily on any single feature (reduces overfitting).
        weight_decay=0.01,
        # Evaluate on validation set after each epoch to track improvement vs overfitting.
        eval_strategy="epoch",
        save_strategy="epoch",
        # After all epochs, reload the checkpoint with the best validation accuracy.
        # The model might peak at epoch 2 and overfit at epoch 3 — this keeps the best version.
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # fp16 mixed precision: use 16-bit floats for forward/backward passes.
        # Halves VRAM usage, faster on modern GPUs (tensor cores), maintains accuracy
        # because weight updates still happen in fp32.
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    # --- Train ---
    # The Trainer wraps the core training loop. For each batch, it does:
    #   1. Forward pass: tokens -> embedding -> 6 transformer layers -> head -> logits
    #   2. Loss: cross-entropy compares logits against true labels.
    #      loss = -log(probability assigned to the correct class)
    #      High confidence in right answer = low loss. Wrong answer = high loss.
    #   3. Backpropagation: starting from the loss, walk backwards through the computation
    #      graph using the chain rule, computing a gradient for every weight (~66M).
    #      The gradient = "how much does the loss change if I nudge this weight?"
    #      Backprop only COMPUTES gradients — it doesn't change any weights.
    #   4. Optimizer step (AdamW): updates every weight individually using its gradient:
    #      w_new = w - learning_rate * gradient
    #      Then clears gradients for the next batch.
    #
    # This repeats for 6,315 batches (2,105 per epoch x 3 epochs).
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...\n")
    trainer.train()

    # --- Evaluate ---
    # Run the model on all validation examples with NO gradient computation (just inference).
    # This measures generalization — how well the model performs on data it never trained on.
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['eval_f1']:.4f}")

    # --- Save ---
    # Save both the fine-tuned model weights (model.safetensors) and the tokenizer config.
    # Both are needed to use the model later — the tokenizer ensures new text is processed
    # with the same vocabulary and rules.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}/")

    # --- Quick inference demo ---
    # pipeline() wraps tokenization + forward pass + softmax into one call.
    # Softmax converts logits into probabilities: [-1.8, 0.3] -> [0.11, 0.89]
    # Returns the label with highest probability and its confidence score.
    print("\n--- Inference Demo ---")
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model=OUTPUT_DIR, device=0 if torch.cuda.is_available() else -1)
    examples = [
        "This movie was absolutely fantastic!",
        "What a terrible waste of time.",
        "It was okay, nothing special.",
        "I loved every minute of it!",
    ]
    for text in examples:
        result = classifier(text)[0]
        label = "Positive" if result["label"] == "LABEL_1" else "Negative"
        print(f"  '{text}' -> {label} ({result['score']:.3f})")


if __name__ == "__main__":
    main()
