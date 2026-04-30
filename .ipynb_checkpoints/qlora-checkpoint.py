# Training script to fine-tune a pre-train LLM with QLoRA using HuggingFace.

import os
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
import evaluate
import numpy as np

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    class DataCollatorForCompletionOnlyLM:
        def __init__(self, response_template_ids, tokenizer):
            self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        def __call__(self, features):
            return self.collator(features)

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with QLoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--model_checkpoints",
        "--model_name",
        dest="model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Checkpoints to path of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id",
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--max_train_retain_samples", type=int, default=None, help="Optional cap for train_retain"
    )
    parser.add_argument(
        "--max_train_forget_samples", type=int, default=None, help="Optional cap for train_forget"
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None, help="Optional cap for each test split"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=64, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments

def get_lora_model(model_checkpoints, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    quantization_config = None
    device_map = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        device_map = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoints,
            device_map=device_map,
            use_safetensors=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoints,
            device_map=device_map,
            use_safetensors=True,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_checkpoints == 'mistralai/Mistral-7B-v0.1' or model_checkpoints == 'meta-llama/Llama-2-7b-hf':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )

    return model, tokenizer, peft_config


def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
    max_train_retain_samples=None,
    max_train_forget_samples=None,
    max_test_samples=None,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return tokenizer(
            prompt_template(examples['text'], examples['label_text']),
            truncation=truncation,
            max_length=max_length,
        )

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    if max_train_retain_samples is not None:
        data['train_retain'] = data['train_retain'].select(range(min(max_train_retain_samples, data['train_retain'].num_rows)))
    if max_train_forget_samples is not None:
        data['train_forget'] = data['train_forget'].select(range(min(max_train_forget_samples, data['train_forget'].num_rows)))
    if max_test_samples is not None:
        data['test_retain'] = data['test_retain'].select(range(min(max_test_samples, data['test_retain'].num_rows)))
        data['test_forget'] = data['test_forget'].select(range(min(max_test_samples, data['test_forget'].num_rows)))

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['text', 'label', 'label_text'])
    data.set_format("torch")

    print(data)

    return data, data_collator


def main(args):
    model_name = args.model_checkpoints.split('/')[-1].lower().replace('.', '-')
    
    # Sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ["WANDB_PROJECT"] = f'qlora_{model_name.lower()}_{args.dataset.lower()}'  # log to your project
    
    data_path = get_data_path(args.dataset)

    if args.output_path is None:
        args.output_path = f'qlora_checkpoints/{model_name.lower()}-hf-qlora-{args.dataset.lower()}'

        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

    # Initialize models and collator
    model, tokenizer, lora_config = get_lora_model(
        args.model_checkpoints,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path,
        tokenizer=tokenizer,
        max_train_retain_samples=args.max_train_retain_samples,
        max_train_forget_samples=args.max_train_forget_samples,
        max_test_samples=args.max_test_samples,
        max_length=args.max_length,
        add_prefix_space=True,
        truncation=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        run_name=f'lr={args.lr}',
        max_grad_norm=0.3,
        metric_for_best_model=None,
    )

    summary(model)

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if torch.cuda.is_available() and model.device.type != 'cuda':
        model = model.to('cuda')

    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=concatenate_datasets([dataset['train_retain'], dataset['train_forget']]),
        eval_dataset={"test": concatenate_datasets([dataset['test_retain'], dataset['test_forget']])},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    runtime = (time.perf_counter()-start)
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)
