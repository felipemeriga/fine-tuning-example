# Fine-Tuning DistilBERT for Sentiment Analysis

An educational example of fine-tuning a pre-trained language model for binary sentiment classification, using Hugging Face `transformers` and `datasets`.

## What this does

Takes **DistilBERT** (a 66M parameter model pre-trained on Wikipedia + BookCorpus) and fine-tunes it on the **SST-2** dataset (67K movie review sentences labeled positive/negative). After ~8 minutes of training on a single GPU, the model achieves **~90% accuracy**.

## The code is heavily commented

The script `fine_tune_sentiment.py` is designed as a learning resource. Every section includes detailed comments explaining:

- **The full fine-tuning pipeline** — tokenization, embeddings, transformer layers, classification head
- **How training works** — forward pass, cross-entropy loss, backpropagation, optimizer step
- **Self-attention** — Q/K/V mechanism, attention scores, how context flows between tokens
- **Softmax used twice** — inside attention (internal mixing) vs at the output (actual prediction)
- **Classification vs LLM training** — bidirectional vs causal attention, supervised labels vs next-token prediction
- **LLM training vs inference** — parallel sequence processing vs token-by-token generation, KV cache, sampling strategies

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA support (tested on RTX 4060 8GB)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup and run

```bash
# Clone the repo
git clone https://github.com/felipemeriga/fine-tuning-example.git
cd fine-tuning-example

# Install dependencies (includes PyTorch with CUDA 12.8)
uv sync

# Run the fine-tuning script
uv run python fine_tune_sentiment.py
```

## Results

| Metric   | Score  |
|----------|--------|
| Accuracy | 90.02% |
| F1 Score | 90.26% |

Inference examples:

```
'This movie was absolutely fantastic!' -> Positive (0.998)
'What a terrible waste of time.'       -> Negative (0.996)
'It was okay, nothing special.'        -> Negative (0.766)
'I loved every minute of it!'          -> Positive (0.996)
```

## API server

After training, serve the model as a REST API:

```bash
uv run uvicorn api:app --reload
```

### Endpoints

**Single prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

```json
{"text": "This movie was absolutely fantastic!", "label": "Positive", "score": 0.9976}
```

**Batch prediction:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Terrible film", "I loved it"]}'
```

```json
{"predictions": [
  {"text": "Terrible film", "label": "Negative", "score": 0.9948},
  {"text": "I loved it", "label": "Positive", "score": 0.9966}
]}
```

**Health check:**

```bash
curl http://localhost:8000/health
```

Interactive docs are available at http://localhost:8000/docs (Swagger UI).

## Project structure

```
fine_tune_sentiment.py  # Training script (heavily commented)
api.py                  # FastAPI inference server
notes.md                # General notes on how fine-tuning works
pyproject.toml          # uv project config with dependencies
uv.lock                 # Locked dependency versions
```
