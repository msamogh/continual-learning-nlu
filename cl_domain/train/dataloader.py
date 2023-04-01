from typing import *

import torch
import torch.utils.data as data
from datasets import Dataset

from transformers import T5Tokenizer

def get_dataloader(args, cl_run_input: "CLRunInput", domain_name: Text, split: Text, tokenizer):
    # For domain "domain_name", get the train, val, and test splits.
    all_samples = cl_run_input.domain_wise_splits[domain_name]

    # We use train_utterances, val_utterances, and test_utterances.
    if split == "train":
        split_samples = all_samples.train_samples
    elif split == "val":
        split_samples = all_samples.val_samples
    elif split == "test":
        split_samples = all_samples.test_samples
    else:
        raise ValueError(f"Invalid split {split}.")

    tokenized_samples = [{
        "input_ids": tokenizer(sample.model_input, padding="max_length", truncation=True,  max_length=args["input_max_length"], return_tensors="pt")["input_ids"][0],
        "labels": tokenizer(sample.model_output, padding="max_length", truncation=True,  max_length=args["input_max_length"], return_tensors="pt")["input_ids"][0],
    } for sample in split_samples]

    dataset = Dataset.from_list(tokenized_samples)
    return dataset
