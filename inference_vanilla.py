# Script to run inference with Vanilla LLM (no fine-tuning)

import os
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from utils import get_data_path


def get_args():
    parser = ArgumentParser(description="Inference with vanilla pretrained LLM")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--set_pad_id", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_json", type=str, default="inference_vanilla_results.json")
    return parser.parse_args()


def compute_metrics(predictions, labels, prefix):
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted', zero_division=0)
    precision = precision_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    return {
        f'{prefix}_accuracy': accuracy, f'{prefix}_f1': f1,
        f'{prefix}_precision': precision, f'{prefix}_recall': recall
    }


def get_model(model_checkpoints, max_length=1024):
    # FIX: use 4-bit quantization to avoid OOM
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoints,
        device_map="auto",
        quantization_config=quantization_config,
        offload_folder="offload",
        trust_remote_code=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=1,
        do_sample=True,
        top_k=1,
        eos_token_id=model.config.eos_token_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints, truncation=True, padding=True, max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer, generation_config


def get_dataset(data_path):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment:"""

    def _preprocessing(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    data = load_dataset(data_path)
    data = data.map(_preprocessing, batched=False)
    data = data.remove_columns(['label_text'])
    data.set_format("torch")
    print(data)
    return data


def inference(model, tokenizer, generation_config, data, max_length=1024):
    device = next(model.parameters()).device
    inputs = tokenizer(
        data['text'], truncation=True, padding=True,
        max_length=max_length, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        token_outputs = model.generate(**inputs, generation_config=generation_config)

    decoded = tokenizer.batch_decode(token_outputs, skip_special_tokens=True)
    predictions = [out[len(text):] for out, text in zip(decoded, data['text'])]

    label_map = {0: 'negative', 1: 'positive'}
    int_predictions = []
    for pred in predictions:
        if 'negative' in pred.lower():
            int_predictions.append(0)
        elif 'positive' in pred.lower():
            int_predictions.append(1)
        else:
            int_predictions.append(2)

    return int_predictions


def batched_inference(model, tokenizer, generation_config, data, batch_size, prefix, max_length):
    all_predictions = []
    for start in tqdm(range(0, data.num_rows, batch_size), desc=prefix):
        batch = data[start: start + batch_size]
        preds = inference(model, tokenizer, generation_config, batch, max_length)
        all_predictions.extend(preds)

    metrics = compute_metrics(all_predictions, data['label'], prefix)
    return metrics


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    # Map model name to HuggingFace path
    name_lower = args.model_name.lower()
    if 'llama-2-7b' in name_lower:
        model_checkpoints = 'meta-llama/Llama-2-7b-hf'
    elif 'llama-2-13b' in name_lower:
        model_checkpoints = 'meta-llama/Llama-2-13b-hf'
    elif 'opt-1.3b' in name_lower:
        model_checkpoints = 'facebook/opt-1.3b'
    else:
        model_checkpoints = args.model_name

    data_path = get_data_path(args.dataset)
    model, tokenizer, generation_config = get_model(model_checkpoints, max_length=args.max_length)
    dataset = get_dataset(data_path)

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id
        generation_config.pad_token_id = model.config.eos_token_id

    all_results = {}
    for prefix in ['train_retain', 'train_forget', 'test_retain', 'test_forget']:
        print(f"\n=== Evaluating {prefix} ===")
        metrics = batched_inference(
            model, tokenizer, generation_config,
            dataset[prefix], args.batch_size, prefix, args.max_length
        )
        all_results[prefix] = metrics
        print(metrics)

    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== FINAL RESULTS ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    args = get_args()
    main(args)
