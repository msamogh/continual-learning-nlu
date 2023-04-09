from typing import *

import numpy as np
import torch
from datasets import Dataset
from transformers import T5Tokenizer, \
    T5ForConditionalGeneration, Trainer, TrainingArguments

from cl_domain.evaluation import create_compute_metrics
from cl_domain.train.dataloader import get_dataloader

MODEL_NAME = "t5-base"
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)


def get_training_args(args: Dict[Text, Any]):
    return TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=50,
        save_total_limit=1,
        learning_rate=1e-4,
    )


def continually_train(args: Dict[Text, Any], training_args: TrainingArguments, cl_run_input: "CLRunInput") -> None:
    print(f"PHASE 1: Training on all domains.")

    for domain_idx, (domain, domain_wise_dataloader) in enumerate(
            cl_run_input.get_ordered_dataloaders(args)
    ):
        print(f"Training {domain.domain_name}.")
        train_dl, val_dl, test_dl = (
            domain_wise_dataloader["train"],
            domain_wise_dataloader["val"],
            domain_wise_dataloader["test"]
        )

        # If domain_idx == 0, then we are training on the first domain.
        # Else load the checkpoint from the previous domain.
        if domain_idx == 0:
            model = MODEL
        else:
            model = T5ForConditionalGeneration.from_pretrained(f"../cl_checkpoints/{args['cl_super_run_label']}/{cl_run_input.label}/after_{domain_idx - 1}")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dl,
            eval_dataset=val_dl,
        )
        trainer.train()

        # Save checkpoint
        model.save_pretrained(f"../cl_checkpoints/{args['cl_super_run_label']}/{cl_run_input.label}/after_{domain_idx}")