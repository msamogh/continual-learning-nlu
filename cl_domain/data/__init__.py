import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

from sklearn.model_selection import train_test_split

from cl_domain.utils import GLOBAL_RAND


@dataclass(frozen=True)
class Sample:
    """A single utterance represented as a (text, intent) pair."""
    text: Text
    intent_label: Text
    domain: Text


@dataclass(frozen=True)
class Domain:
    """A domain is a collection of utterances that are related to each other.

    For example, the data "weather" might contain utterances such as "What's
    the weather like today?" and "What's the weather like in New York?".
    """
    dataset: Text
    domain: Text
    utterances: Dict[Text, List[Sample]] = field(default_factory=lambda: defaultdict(list))

    @staticmethod
    def populate_domains():
        all_domains = {}
        # Every dataset
        for dataset in ("multiwoz", "sgd", "tm_2019", "tm_2020"):
            # Every split
            for split in ("train", "valid", "test"):
                # Every file a split
                with (Path("../../data") / dataset / f"{split}.json").open("r") as f:
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
                                        dialogue["dialogue"][idx - 1][
                                            "spk"] == "USER" and
                                        "(" in turn["utt"]
                                ):
                                    all_domains[domain].utterances[split].append(
                                        Sample(
                                            dialogue["dialogue"][idx - 1][
                                                "utt"],
                                            turn["utt"][:turn["utt"].index("(")],
                                            domain
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
    test_utterances: List[Sample]

    @classmethod
    def sample_from_domain(cls, domain: Domain,
                           test_size: float = 0.2):
        train_utterances, test_utterances = train_test_split(
            domain.utterances,
            test_size=test_size,
            random_state=GLOBAL_RAND
        )
        return cls(domain, train_utterances, test_utterances)


if __name__ == "__main__":
    all_domains = Domain.populate_domains()
    print("done")
