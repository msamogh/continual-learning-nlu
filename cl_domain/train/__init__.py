from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, Trainer

from cl_domain.experiment import CLRunInput
from cl_domain.train.dataloader import get_dataloader


@dataclass
class CLTrainConfig:
    model_name: str


def get_trainer(cl_run_input: CLRunInput, model_config: CLTrainConfig):
    model = T5ForConditionalGeneration.from_pretrained(model_config.model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_config.model_name)
    train_dl, val_dl, test_dl = (
        get_dataloader(cl_run_input, "train"),
        get_dataloader(cl_run_input, "val"),
        get_dataloader(cl_run_input, "test")
    )
    return Trainer(
        model
    )
