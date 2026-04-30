# Script to run inference using SPUL

import os
import pickle
import datetime
import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    class DataCollatorForCompletionOnlyLM:
        def __init__(self, response_template_ids, tokenizer):
            self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        def __call__(self, features):
            return self.collator(features)

from utils import get_data_path, preprocess_logits_for_metrics, CustomCallback
from spul import get_unlearn_dataset_and_collator, get_unlearning_loss_trainer

import evaluate as hf_evaluate
from sklearn.metrics import accuracy_score, f1_score

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="Run inference using SPUL")
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--model_checkpoints", type=str, default=None, required=True)
    parser.add_argument("--logits_path", type=str, default=None, required=False)
    parser.add_argument("--forget_size", type=float, default=1.0, required=False)
    parser.add_argument("--output_path", type=str, default=None, required=False)
    return parser.parse_args()


def compute_metrics_spul(eval_pred, num_virtual_tokens=30):
    """Fixed compute_metrics that handles virtual prompt token offsets."""
    logits, labels = eval_pred

    # logits has extra virtual token dims at the front — strip them
    if logits.shape[1] > labels.shape[1]:
        offset = logits.shape[1] - labels.shape[1]
        predictions = logits[:, offset-1:-1]
    else:
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


def get_ptuning_model(model_checkpoints, lora_checkpoints, max_length):
    lora_config = PeftConfig.from_pretrained(lora_checkpoints)
    base_model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        device_map="auto", offload_folder="offload", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path,
        truncation=True, padding=True, max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoints)
    lora_model = lora_model.merge_and_unload()
    model = PeftModel.from_pretrained(lora_model, model_checkpoints)
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    model_name = args.model_checkpoints.split('/')[-1].lower().replace('.', '-')
    data_path = get_data_path(args.dataset)

    if args.logits_path is None:
        args.logits_path = f'saved_logits/llama-2-7b-hf-hf-qlora-sst2_sst2-1.0.pkl'

    # Load arguments from saved model
    path = args.model_checkpoints
    with open(os.path.join(path, 'arguments.txt'), 'r') as f:
        parameters = f.readlines()
    params = {}
    for line in parameters:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            params[parts[0].strip()] = parts[1].strip()

    model, tokenizer = get_ptuning_model(
        args.model_checkpoints,
        params.get('model_checkpoints', params.get('model_name')),
        int(params['max_length'])
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path, tokenizer=tokenizer,
        max_length=int(params['max_length']),
        add_prefix_space=True, truncation=True,
    )

    with open(args.logits_path, 'rb') as f:
        original_logits = pickle.load(f)

    if args.output_path is None:
        args.output_path = os.path.join(path, "inference_outputs")

    num_virtual_tokens = int(params.get('ptuning_num_tokens', 30))

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=float(params['lr']),
        per_device_eval_batch_size=int(params['eval_batch_size']),
        num_train_epochs=int(params['num_epochs']),
        weight_decay=float(params['weight_decay']),
        evaluation_strategy="no",
        save_strategy="no",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    if params.get('set_pad_id') == 'True':
        model.config.pad_token_id = model.config.eos_token_id

    if torch.cuda.is_available() and model.device.type != 'cuda':
        model = model.to('cuda')

    custom_loss = get_unlearning_loss_trainer()

    # wrap compute_metrics to handle virtual token offset
    def _compute_metrics(eval_pred):
        return compute_metrics_spul(eval_pred, num_virtual_tokens=num_virtual_tokens)

    trainer = custom_loss(
        model=model,
        original_logits=original_logits,
        num_virtual_tokens=num_virtual_tokens,
        alpha=float(params['alpha']),
        beta=float(params['beta']),
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset={
            "train_retain": dataset['train_retain'],
            "train_forget": dataset['train_forget'],
            "test_retain": dataset['test_retain'],
            "test_forget": dataset['test_forget'],
        },
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=_compute_metrics,
    )

    print("\n=== Evaluating SPUL on all splits ===")
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
