import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

from sklearn.model_selection import train_test_split



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
    dataset_name: Text
    domain_name: Text
    splits: Dict[Text, List[Sample]] = field(
        default_factory=lambda: defaultdict(list))

    @staticmethod
    def generate_samples(ctx_window_size: Optional[int] = None) -> Dict[Text, "Domain"]:
        domain_wise_samples = {}
        # Every dataset
        for dataset in ("sgd",):  # ("multiwoz", "sgd", "tm_2019", "tm_2020"):
            # Every split in a dataset
            for split in ("train", "valid", "test"):
                # Every file in a split
                with (Path("./data") / dataset / f"{split}.json").open(
                        "r") as f:
                    dialogues = json.load(f)
                    # Ever dialogue in a file
                    for dialogue in dialogues:
                        # Only consider dialogues with a single domain
                        domains = dialogue["services"]
                        if len(domains) > 1:
                            continue
                        domain = domains[0]
                        if domain not in domain_wise_samples:
                            domain_wise_samples[domain] = Domain(dataset,
                                                                 domain)

                        # Every turn in a dialogue
                        for idx, turn in enumerate(dialogue["dialogue"]):
                            if turn["spk"] == "API":
                                # If immediately preceded by a user turn
                                if (
                                        idx > 0 and
                                        dialogue["dialogue"][idx - 1][
                                            "spk"] == "USER" and
                                        "(" in turn["utt"]
                                ):
                                    # Consider all utterances within a fixed window as the context.
                                    if ctx_window_size is None:
                                        context = [
                                            Turn(speaker=turn["spk"],
                                                 utterance=turn["utt"])
                                            for turn in
                                            dialogue["dialogue"][:idx]
                                        ]
                                    else:
                                        context = [
                                            Turn(speaker=turn["spk"],
                                                 utterance=turn["utt"])
                                            for turn in dialogue["dialogue"][
                                                        idx - ctx_window_size + 1:idx]
                                        ]
                                    domain_wise_samples[domain].splits[
                                        split].append(
                                        Sample(
                                            context=context,
                                            intent_label=turn["utt"][
                                                         :turn["utt"].index(
                                                             "(")],
                                            domain=domain
                                        )
                                    )
        return domain_wise_samples


@dataclass(frozen=True)
class DomainSplit:
    """A subset of a data.

    DomainSubset objects are used to represent a subset of a data. This is
    useful for evaluating the performance of a train on a subset of a data.
    """
    domain: Domain
    train_samples: List[Sample]
    val_samples: List[Sample]
    test_samples: List[Sample]

    @classmethod
    def get_fixed_n_split(cls, domain: Domain, n: Union[int, float], test_size: float, val_size: float) -> "DomainSplit":
        """First, combine the train, val, and test splits. Then, split the
        combined splits into train and test splits. Finally, split the train
        split into train and val splits."""
        if isinstance(n, float):
            n = int(len(domain.splits) // n)

        train_utterances = domain.splits["train"]
        val_utterances = domain.splits["valid"]
        test_utterances = domain.splits["test"]
        combined_utterances = train_utterances + val_utterances + test_utterances

        train_utterances, test_utterances = train_test_split(
            combined_utterances, test_size=test_size, random_state=42)
        train_utterances, val_utterances = train_test_split(train_utterances, test_size=val_size / (1 - test_size), random_state=42)

        return cls(domain, train_utterances, val_utterances, test_utterances)

