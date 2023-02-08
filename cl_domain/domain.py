import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

from sklearn.model_selection import train_test_split

from cl_domain.config import get_args
from cl_domain.utils import GLOBAL_RAND


@dataclass
class Turn:
    speaker: Text
    utterance: Text


@dataclass(frozen=True)
class Sample:
    """A single utterance represented as a (text, intent) pair."""
    context: List[Turn]
    intent_label: Text
    domain: Text

    task_prefix: Text = ""
    sep_token: Text = " </s> "

    @property
    def model_input(self):
        tokenized = ""
        for turn in self.context:
            tokenized += f"{turn.speaker}: {turn.utterance} {self.sep_token} "
        tokenized = f'{self.task_prefix} {tokenized}'
        tokenized = tokenized.strip().replace("  ", " ")
        return tokenized

    @property
    def model_output(self):
        return f"{self.intent_label} {self.sep_token}"


@dataclass(frozen=True)
class Domain:
    """A domain is a collection of utterances that are related to each other.

    For example, the data "weather" might contain utterances such as "What's
    the weather like today?" and "What's the weather like in New York?".
    """
    dataset: Text
    domain: Text
    utterances: Dict[Text, List[Sample]] = field(
        default_factory=lambda: defaultdict(list))

    @staticmethod
    def get_all_domains(ctx_window_size: int):
        all_domains = {}
        # Every dataset
        for dataset in ("multiwoz", "sgd", "tm_2019", "tm_2020"):
            # Every split in a dataset
            for split in ("train", "valid", "test"):
                # Every file in a split
                with (Path("data") / dataset / f"{split}.json").open(
                        "r") as f:
                    dialogues = json.load(f)
                    # Ever dialogue in a file
                    for dialogue in dialogues:
                        # Only consider dialogues with a single domain
                        domains = dialogue["services"]
                        if len(domains) > 1:
                            continue
                        domain = domains[0]
                        if domain not in all_domains:
                            all_domains[domain] = Domain(dataset, domain)

                        # Every turn in a dialogue
                        for idx, turn in enumerate(dialogue["dialogue"]):
                            if turn["spk"] == "API":
                                # If immediately preceded by a user turn
                                if (
                                    idx > 0 and
                                    dialogue["dialogue"][idx - 1]["spk"] == "USER" and
                                    "(" in turn["utt"]
                                ):
                                    # Consider all utterances within a fixed window as the context.
                                    context = [
                                        Turn(speaker=turn["spk"], utterance=turn["utt"])
                                        for turn in dialogue["dialogue"][idx - ctx_window_size + 1:idx]
                                    ]
                                    all_domains[domain].utterances[split].append(
                                        Sample(
                                            context=context,
                                            intent_label=turn["utt"][:turn["utt"].index("(")],
                                            domain=domain
                                        )
                                    )
        return all_domains


@dataclass(frozen=True)
class DomainSplit:
    """A subset of a data.

    DomainSubset objects are used to represent a subset of a data. This is
    useful for evaluating the performance of a model on a subset of a data.
    """
    domain: Domain
    train_utterances: List[Sample]
    val_utterances: List[Sample]
    test_utterances: List[Sample]

    @classmethod
    def subsample(cls, domain: Domain, n: Union[int, float], test_size: float = 0.2):
        if isinstance(n, float):
            n = int(len(domain.utterances) // n)
        train_utterances = domain.utterances["train"][:n]
        val_utterances = domain.utterances["valid"]
        test_utterances = domain.utterances["test"]
        return cls(domain, train_utterances, val_utterances, test_utterances)


if __name__ == "__main__":
    args = get_args()
    print("done")
