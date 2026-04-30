import os
import pickle
import datetime
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numpy.random import default_rng
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback

def get_data_path(dataset):
    dataset_name = dataset.lower()
    if dataset_name == "sst2":
        data_path = "karuna-bhaila/Unlearning_SST2"
    elif dataset_name == 'yelp':
        data_path = "karuna-bhaila/Unlearning_Yelp_Polarity"
    else:
        raise NotImplementedError
    return data_path


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def get_logits_from_base_model(base_model, data_collator, dataset):
    # batch_size=1 to minimize memory pressure during logit extraction
    train_loader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=1)
    original_logits = {}

    device = next(base_model.parameters()).device

    progress_bar = tqdm(train_loader)
    for sample in progress_bar:
        sample.pop('is_forget')
        indices = sample.pop('index')

        # move tensors to model device
        sample = {k: v.to(device) for k, v in sample.items()}

        with torch.no_grad():
            logits = base_model(**sample).get('logits')

        attention_mask = sample.get('attention_mask')
        for i in range(logits.shape[0]):
            seq_len = int(attention_mask[i].sum().item())
            # store on CPU immediately to free GPU memory
            original_logits[indices[i]] = logits[i, seq_len - 1].cpu()

        # FIX: clear GPU cache after every batch to prevent OOM buildup
        del logits
        torch.cuda.empty_cache()

    return original_logits


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')

    logits, labels = eval_pred

    predictions = logits[:, :-1]
    labels = labels[:, 1:]

    check_labels = labels != -100

    last_token_predictions = []
    last_token_labels = []

    for idx in range(len(predictions)):
        last_token_predictions.append(predictions[idx][check_labels[idx]])
        last_token_labels.append(labels[idx][check_labels[idx]])

    f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
    precision = precision_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['precision']
    recall = recall_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['recall']
    return {"f1-score": f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retrain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy
