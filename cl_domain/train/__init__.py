import json
import random
from typing import *
from collections import deque

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments, EarlyStoppingCallback,
)

from cl_domain.evaluation import create_compute_metrics
from cl_domain.train.dataloader import get_dataloader

MODEL_NAME = "t5-base"
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)


def get_training_args(args: Dict[Text, Any], cl_step_idx: int) -> TrainingArguments:
    # Set learning rate inversely proportional to the number of steps.
    if args["cl_lr_schedule"] == "linear":
        learning_rate = args["lr"] / (2 * cl_step_idx + 1)
    elif args["cl_lr_schedule"] == "constant":
        learning_rate = args["lr"]
    else:
        raise ValueError(f"Invalid learning rate schedule {args['cl_lr_schedule']}.")

    deepspeed_config = json.load(open(args["deepspeed_config"], "r"))
    # print(deepspeed_config)

    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=args["num_train_epochs"],
        per_device_train_batch_size=args["train_batch_size"],
        per_device_eval_batch_size=args["eval_batch_size"],
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        gradient_accumulation_steps=1,
        logging_steps=1,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        eval_steps=3,
        fp16=True,
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=1,
        learning_rate=learning_rate,
    )


def continually_train(
    args: Dict[Text, Any],
    training_args: TrainingArguments,
    cl_run_input: "CLRunInput",
) -> None:
    print(f"PHASE 1: Training on all domains.")

    for domain_idx, (domain, domain_wise_dataloader) in enumerate(
        cl_run_input.get_ordered_dataloaders(args)
    ):
        print(f"Training {domain.domain_name}.")
        train_dl, val_dl, test_dl = (
            domain_wise_dataloader["train"],
            domain_wise_dataloader["val"],
            domain_wise_dataloader["test"],
        )

        if domain_idx > 0:
            for replay_domain_idx in range(domain_idx):
                print(
                    f"Concatenating experience replay data from domain name: {cl_run_input.domain_ordering[replay_domain_idx].domain_name}."
                )
                prev_train_dl = get_dataloader(
                    args,
                    cl_run_input,
                    cl_run_input.domain_ordering[replay_domain_idx].domain_name,
                    "train",
                    TOKENIZER,
                    subsample_size=args["cl_experience_replay_size"],
                )
                train_dl = concatenate_datasets([train_dl, prev_train_dl])

        # If domain_idx == 0, then we are training on the first domain.
        # Else load the checkpoint from the previous domain.
        if domain_idx == 0:
            model = MODEL
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                f"../cl_checkpoints/{args['cl_super_run_label']}/{cl_run_input.label}/after_{domain_idx - 1}"
            )

        # Train
        print(f"Validation length: {len(val_dl)}")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dl,
            eval_dataset=val_dl,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0
            )]
        )
        trainer.train()

        # Save checkpoint
        model.save_pretrained(
            f"{args['cl_checkpoint_dir']}/{args['cl_super_run_label']}/{cl_run_input.label}/after_{domain_idx}"
        )
