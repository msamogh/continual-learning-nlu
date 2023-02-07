from dataclasses import dataclass
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
    domain: Text
    utterances: List[Sample]
    dataset: Optional[Text] = None


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
