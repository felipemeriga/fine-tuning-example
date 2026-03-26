# How Fine-Tuning a Language Model Works

## The Big Idea

A pre-trained model has already learned language structure, grammar, and semantics from billions of words. Fine-tuning takes that existing knowledge and adapts it to a specific task (classification, summarization, Q&A, etc.) using a much smaller labeled dataset. Instead of months of training from scratch, it takes minutes.

## The Pipeline

### 1. Tokenization

Neural networks need numbers, not text. A tokenizer splits text into subword units and maps each to an integer ID using a fixed vocabulary. "unhappiness" might become `["un", "##happi", "##ness"]` -> `[4895, 18223, 4757]`. Special tokens are added: `[CLS]` at the start (used for classification) and `[SEP]` at the end. Sequences are padded to equal length, and an attention mask marks which tokens are real vs padding.

### 2. Embedding

The first layer of the model is a learned lookup table. Each token ID maps to a dense vector (e.g., 768 dimensions). Position embeddings are added so the model knows word order. This layer is part of the model — its weights are pre-trained and continue to be updated during fine-tuning.

### 3. Transformer Layers

Each layer applies **self-attention**: every token computes how relevant every other token is to it, using Query, Key, and Value projections. This is what lets the model understand context — connecting "great" to "movie" across a sentence. Each layer also has a feed-forward network. Residual connections (skip connections) prevent gradients from vanishing in deep networks.

### 4. Task Head

A small neural network added on top of the pre-trained body. For classification, it takes the `[CLS]` token's final vector and outputs one raw number (logit) per class. This head is randomly initialized — it's the part that needs the most learning.

### 5. Loss Computation

The logits (raw unbounded numbers) are converted to probabilities via softmax. Cross-entropy loss measures how far the prediction is from the true label:

```
loss = -log(probability assigned to the correct class)
```4

High confidence in the right answer = low loss. Wrong answer = high loss. The batch loss is the average across all examples in the batch.

### 6. Backpropagation

Starting from the loss, PyTorch walks backwards through the computation graph using the chain rule, computing a gradient for every weight in the model. The gradient tells us: "if I nudge this weight, how much does the loss change, and in which direction?" Backpropagation only computes gradients — it doesn't change any weights.

### 7. Optimizer Step

The optimizer (typically AdamW) uses the gradients to update every weight individually:

```
weight_new = weight - learning_rate x gradient
```

The learning rate is kept very small (e.g., 2e-5) to avoid destroying pre-trained knowledge — a problem called catastrophic forgetting.

### 8. Repeat

Steps 1-7 repeat for every batch, for every epoch. Over thousands of iterations, the loss decreases and the model learns the task.

## The Fundamental Building Block

Every operation inside the network — attention, feed-forward layers, the classification head — is built from the same primitive:

```
output = activation(W x input + bias)
```

A linear transformation followed by a non-linear activation function (ReLU, GELU, etc.). The activation is what lets neural networks learn complex patterns — without it, stacking layers would collapse into a single linear operation.

## Why Fine-Tuning Works

Training a model from scratch on a small dataset would fail — not enough data to learn language from nothing. Fine-tuning works because the pre-trained weights already encode deep language understanding. We just teach the model to apply that understanding to our specific task, which requires far less data and compute.
