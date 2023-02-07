from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration


@dataclass
class CLModel:
    model_name: str
    model: AutoModel
    tokenizer: AutoTokenizer

    def __post_init__(self):
        if "t5" in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError("Model type not supported")
