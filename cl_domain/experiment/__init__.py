from typing import *
from dataclasses import dataclass

from cl_domain.domain import *
from cl_domain.experiment.ordering import random_ordering, max_path_ordering, \
    min_path_ordering
from cl_domain.train import CLTrainConfig


ORDERINGS = {
    "random": random_ordering,
    "max_path": max_path_ordering,
    "min_path": min_path_ordering,
}


@dataclass(frozen=True)
class CLRunInput:
    domain_ordering: List[Domain]
    domain_wise_samples: Dict[Text, DomainSplit]


@dataclass(frozen=True)
class CLRunResult:
    cl_run_input: CLRunInput
    cl_run_accuracies: List[Dict[Text, float]]
