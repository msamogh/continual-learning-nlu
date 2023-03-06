from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, Trainer

from cl_domain.experiment import CLRunInput
from cl_domain.train.dataloader import get_dataloader


MODEL_NAME = "t5-small"


def init_trainer(cl_run_input: CLRunInput):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    return Trainer(
        model
    )

def continually_train(trainer: Trainer, cl_run_input: CLRunInput):
    for domain in cl_run_input.domain_ordering:
        train_dl, val_dl, test_dl = (
            get_dataloader(cl_run_input, domain.domain_name, "train"),
            get_dataloader(cl_run_input, domain.domain_name, "val"),
            get_dataloader(cl_run_input, domain.domain_name, "test")
        )
        trainer.fit(train_dl, val_dl)
        metrics = trainer.evaluate(test_dl)
