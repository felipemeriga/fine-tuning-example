# CSV Bank Transaction Normalizer — Implementation Plan

## Goal

Build a system that takes CSV exports from multiple Brazilian banks and normalizes
them into a standard schema. Code handles the deterministic pipeline (reading rows,
validation, saving). A fine-tuned local model handles the intelligent part
(merchant normalization, category classification, ambiguous date parsing).

## Architecture

```
CSV file (any bank format)
    |
    v
Python reads CSV (deterministic — every row guaranteed)
    |
    v
For each row, the model normalizes:
    - Merchant name: "RCHLO*Riachuelo" -> "Riachuelo"
    - Category: "UBER *TRIP BR" -> "Transport"
    - Date: "15/Mar" or "03-15" -> "2024-03-15"
    |
    v
Output: standardized JSON/CSV with schema:
    {"date": "2024-03-15", "description": "Uber Trip", "amount": -25.90, "category": "Transport"}
```

## Output Schema

Each transaction normalized to:

| Field       | Type   | Example           |
|-------------|--------|-------------------|
| date        | string | "2024-03-15"      |
| merchant    | string | "Riachuelo"       |
| description | string | "RCHLO*Riachuelo" |
| amount      | float  | -159.90           |
| category    | string | "Shopping"        |

Categories: Transport, Food, Shopping, Transfer, Bills, Entertainment,
Health, Education, Subscriptions, Other.

## Steps

### 1. Collect and label data

- Gather CSVs from different banks (Nubank, Itau, Santander, Bradesco, Inter, etc.)
- Each bank has different column names, delimiters, date formats, encodings
- Manually label ~50-100 real transactions with the target schema
- This is the most time-consuming step — but critical for quality

### 2. Synthetic data generation

- From the real labeled data, generate ~1,000-2,000 synthetic examples
- Randomize: dates, amounts, merchant names, descriptions
- Vary: column names, delimiters (comma, semicolon, tab), date formats,
  header presence, column ordering, encoding (UTF-8, Latin-1)
- Add noise: extra whitespace, missing fields, special characters
- This gives the model enough volume to learn generalization

### 3. Format training data

Each training example is a prompt/response pair:

```
Prompt:  "Normalize this bank transaction: Data: 15/03/2024, Descrição: UBER *TRIP SP, Valor: -25.90"
Response: {"date": "2024-03-15", "merchant": "Uber", "description": "UBER *TRIP SP", "amount": -25.90, "category": "Transport"}
```

### 4. Fine-tune with QLoRA

- Base model: TinyLlama 1.1B or LLaMA 3.2 1B (if approved)
- QLoRA: 4-bit quantization + LoRA adapters (same approach as llm/fine_tune_llama.py)
- Train for 3-5 epochs on the synthetic dataset
- Monitor: loss, exact match accuracy per field

### 5. Build the pipeline

```python
# Pseudocode
csv_rows = read_csv(file)          # deterministic — code handles this
for row in csv_rows:
    raw_text = format_row(row)      # convert row to text prompt
    normalized = model(raw_text)    # model normalizes fields
    validated = validate(normalized) # code validates JSON schema
    results.append(validated)
save(results)                       # code saves output
```

Key: the model only normalizes one row at a time. Code handles iteration,
validation, and error handling. No chance of skipping rows.

### 6. Evaluation

- Hold out 1-2 real CSVs (never seen during training) for testing
- Metrics per field:
  - Date: exact match accuracy
  - Merchant: exact match after lowercasing
  - Category: accuracy + confusion matrix (which categories get confused?)
  - Amount: exact match (should be ~100% since it's just number parsing)
- Compare: prompting a large model (Claude/GPT-4) vs fine-tuned local model
  - Accuracy difference
  - Cost per transaction
  - Latency per transaction
  - Privacy considerations (financial data)

### 7. Serve as API

- FastAPI endpoint: POST /normalize with CSV file upload
- Returns normalized JSON array
- Same pattern as the sentiment API (api.py)

## What this demonstrates (for portfolio/interviews)

- **System design**: code for structure + model for intelligence
- **Data engineering**: synthetic data generation from limited real data
- **Fine-tuning**: QLoRA on a practical task
- **Evaluation**: comparing approaches with real metrics
- **Cost analysis**: API costs vs local model
- **Privacy awareness**: financial data stays local
- **Production readiness**: API server, validation, error handling

## Dependencies

Same as the LLM fine-tuning (already in pyproject.toml):
- torch, transformers, peft, bitsandbytes, trl
- Plus: pandas (for CSV handling)

## File structure (planned)

```
csv-normalizer/
    PLAN.md                  # This file
    generate_synthetic.py    # Step 2: generate training data
    fine_tune.py             # Step 4: QLoRA training
    normalize.py             # Step 5: the pipeline
    api.py                   # Step 7: FastAPI server
    evaluate.py              # Step 6: evaluation script
    data/
        raw/                 # Real CSVs from banks (gitignored)
        labeled/             # Manually labeled examples
        synthetic/           # Generated training data
        test/                # Held-out test CSVs
```
