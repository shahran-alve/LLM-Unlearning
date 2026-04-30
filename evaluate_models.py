import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def prompt_template(text: str) -> str:
    return f"### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment:"


def get_quantization_config():
    if torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None


def classify_batch(model, tokenizer, texts: List[str], max_length: int) -> List[int]:
    prompts = [prompt_template(t) for t in texts]
    inputs = tokenizer(
        prompts, truncation=True, padding=True,
        max_length=max_length, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    seq_lens = inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(logits.shape[0], device=device)
    next_logits = logits[batch_idx, seq_lens]

    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    return [1 if row[pos_id] >= row[neg_id] else 0 for row in next_logits]


def eval_split(model, tokenizer, split, batch_size: int, max_length: int) -> Dict[str, float]:
    y_true = split["label"]
    y_pred = []
    for start in range(0, split.num_rows, batch_size):
        batch = split[start: start + batch_size]
        y_pred.extend(classify_batch(model, tokenizer, batch["text"], max_length=max_length))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def load_tokenizer(base_model_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, truncation=True, padding=True, max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(base_model_name: str):
    """Load base model with 4-bit quantization to save GPU memory."""
    quantization_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_qlora_model(qlora_path: str, max_length: int):
    """Load QLoRA fine-tuned model (merged)."""
    qlora_cfg = PeftConfig.from_pretrained(qlora_path)
    base_model = load_base_model(qlora_cfg.base_model_name_or_path)
    tokenizer = load_tokenizer(qlora_cfg.base_model_name_or_path, max_length)
    model = PeftModel.from_pretrained(base_model, qlora_path)
    model.eval()
    return model, tokenizer


def load_baseline_model(path: str, max_length: int):
    """Load a PEFT adapter baseline model."""
    # FIX: baselines are saved as PEFT adapters, not full models
    peft_cfg = PeftConfig.from_pretrained(path)
    base_model = load_base_model(peft_cfg.base_model_name_or_path)
    tokenizer = load_tokenizer(peft_cfg.base_model_name_or_path, max_length)
    model = PeftModel.from_pretrained(base_model, path)
    model.eval()
    return model, tokenizer


def load_spul_model(qlora_path: str, spul_path: str, max_length: int):
    """Load SPUL model: base + QLoRA adapter + SPUL prompt tuning adapter."""
    qlora_model, tokenizer = load_qlora_model(qlora_path, max_length)
    model = PeftModel.from_pretrained(qlora_model, spul_path)
    model.eval()
    return model, tokenizer


def free_model(model):
    """Free GPU memory after evaluating each model."""
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="karuna-bhaila/Unlearning_SST2")
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--qlora_path", type=str, required=True)
    parser.add_argument("--spul_path", type=str, required=True)
    parser.add_argument("--ga_path", type=str, required=True)
    parser.add_argument("--rl_path", type=str, required=True)
    parser.add_argument("--gagd_path", type=str, required=True)
    parser.add_argument("--gakl_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="results_summary.json")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    test_retain = data["test_retain"].select(range(min(args.max_eval_samples, data["test_retain"].num_rows)))
    test_forget = data["test_forget"].select(range(min(args.max_eval_samples, data["test_forget"].num_rows)))

    results = {}

    # Evaluate QLoRA model
    print("\n=== Evaluating QLoRA model ===")
    model, tokenizer = load_qlora_model(args.qlora_path, args.max_length)
    results["qlora"] = {
        "test_retain": eval_split(model, tokenizer, test_retain, args.batch_size, args.max_length),
        "test_forget": eval_split(model, tokenizer, test_forget, args.batch_size, args.max_length),
    }
    free_model(model)

    # Evaluate baselines
    for name, path in [
        ("gradient_ascent", args.ga_path),
        ("random_label", args.rl_path),
        ("gradient_ascent_descent", args.gagd_path),
        ("gradient_ascent_kl", args.gakl_path),
    ]:
        print(f"\n=== Evaluating {name} ===")
        model, tokenizer = load_baseline_model(path, args.max_length)
        results[name] = {
            "test_retain": eval_split(model, tokenizer, test_retain, args.batch_size, args.max_length),
            "test_forget": eval_split(model, tokenizer, test_forget, args.batch_size, args.max_length),
        }
        free_model(model)

    # Evaluate SPUL
    print("\n=== Evaluating SPUL ===")
    model, tokenizer = load_spul_model(args.qlora_path, args.spul_path, args.max_length)
    results["spul"] = {
        "test_retain": eval_split(model, tokenizer, test_retain, args.batch_size, args.max_length),
        "test_forget": eval_split(model, tokenizer, test_forget, args.batch_size, args.max_length),
    }
    free_model(model)

    with open(args.output_json, "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
