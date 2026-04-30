# Script to run inference on base model fine-tuned with QLoRA

import os
import numpy as np
from argparse import ArgumentParser

import torch
from torchinfo import summary
from sklearn.metrics import accuracy_score, f1_score

from datasets import load_dataset, concatenate_datasets
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, \
    DataCollatorForLanguageModeling, Trainer, BitsAndBytesConfig
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    class DataCollatorForCompletionOnlyLM:
        def __init__(self, response_template_ids, tokenizer):
            self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        def __call__(self, features):
            return self.collator(features)

from utils import get_data_path, preprocess_logits_for_metrics


def get_args():
    parser = ArgumentParser(description="Run inference on QLoRA fine-tuned model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_checkpoints", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits[:, :-1]
    labels = labels[:, 1:]
    check_labels = labels != -100

    last_token_predictions = []
    last_token_labels = []
    for idx in range(len(predictions)):
        valid_preds = predictions[idx][check_labels[idx]]
        valid_labels = labels[idx][check_labels[idx]]
        if len(valid_preds) > 0:
            last_token_predictions.append(valid_preds[-1])
            last_token_labels.append(valid_labels[-1])

    if not last_token_predictions:
        return {"accuracy": 0.0, "f1-score": 0.0}

    accuracy = accuracy_score(y_true=last_token_labels, y_pred=last_token_predictions)
    f1 = f1_score(y_true=last_token_labels, y_pred=last_token_predictions, average='weighted', zero_division=0)
    return {"accuracy": accuracy, "f1-score": f1}


def get_lora_model(model_checkpoints, max_length):
    lora_config = PeftConfig.from_pretrained(model_checkpoints)

    # FIX: use 4-bit quantization to avoid OOM
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        offload_folder="offload",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path,
        truncation=True, padding=True, max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(base_model, model_checkpoints)
    model.eval()
    return model, tokenizer


def get_dataset_and_collator(data_path, tokenizer, max_length=1024, truncation=True):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return tokenizer(
            prompt_template(examples['text'], examples['label_text']),
            truncation=truncation, max_length=max_length,
        )

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)
    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['text', 'label', 'label_text'])
    data.set_format("torch")
    print(data)
    return data, data_collator


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    data_path = get_data_path(args.dataset)

    # Load arguments from saved checkpoint
    with open(os.path.join(args.model_checkpoints, 'arguments.txt'), 'r') as f:
        parameters = f.readlines()
    params = {}
    for line in parameters:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            params[parts[0].strip()] = parts[1].strip()

    model, tokenizer = get_lora_model(args.model_checkpoints, max_length=int(params['max_length']))
    dataset, collator = get_dataset_and_collator(data_path, tokenizer, max_length=int(params['max_length']))

    if args.output_path is None:
        args.output_path = os.path.join(args.model_checkpoints, "inference_outputs")

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_eval_batch_size=4,
        evaluation_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    if params.get('set_pad_id') == 'True':
        model.config.pad_token_id = model.config.eos_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("\n=== Evaluating QLoRA Base Model on all splits ===")
    for split_name, split_data in [
        ("train_retain", dataset['train_retain']),
        ("train_forget", dataset['train_forget']),
        ("test_retain", dataset['test_retain']),
        ("test_forget", dataset['test_forget']),
    ]:
        results = trainer.evaluate(eval_dataset=split_data, metric_key_prefix=split_name)
        print(f"{split_name}: {results}")


if __name__ == "__main__":
    args = get_args()
    main(args)
