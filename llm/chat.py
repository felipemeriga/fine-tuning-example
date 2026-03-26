"""
Interactive chat with the fine-tuned TinyLlama model.

Run with:
    uv run python llm/chat.py

Type your messages and press Enter. Type "quit" to exit.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "./llm/llama-chat-finetuned"


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.")
        return

    print(f"Loading model from {ADAPTER_DIR}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Model loaded. Type 'quit' to exit.\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        conversation.append({"role": "user", "content": user_input})

        input_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        conversation.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
