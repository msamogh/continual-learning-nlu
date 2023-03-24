from typing import *

import torch
from datasets import Dataset
from transformers import T5Tokenizer, \
    T5ForConditionalGeneration, Trainer, TrainingArguments

from cl_domain.train.dataloader import get_dataloader

MODEL_NAME = "t5-small"
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)


def get_training_args(args: Dict[Text, Any]):
    return TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=50,
        save_total_limit=1,
    )


def continually_train(args: Dict[Text, Any], training_args: TrainingArguments, cl_run_input: "CLRunInput"):
    for (domain, domain_wise_dataloader) in cl_run_input.get_ordered_dataloaders(args):
        train_dl, val_dl, test_dl = (
            domain_wise_dataloader["train"],
            domain_wise_dataloader["val"],
            domain_wise_dataloader["test"]
        )
        trainer = Trainer(
            model=MODEL,
            args=training_args,
            train_dataset=train_dl,
            eval_dataset=val_dl,
            # data_collator=lambda data: {
            #     'input_ids': torch.stack([f[0] for f in data]),
            #     'attention_mask': torch.stack([f[1] for f in data]),
            #     'labels': torch.stack([f[0] for f in data])
            # }
        )
        trainer.train()
        metrics = trainer.evaluate(test_dl)
