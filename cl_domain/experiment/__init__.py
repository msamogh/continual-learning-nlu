from typing import *

from cl_domain.domain import *
from cl_domain.experiment.ordering import random_ordering, max_path_ordering, \
    min_path_ordering
from cl_domain.model import CLModel


ORDERINGS = {
    "random": random_ordering,
    "max_path": max_path_ordering,
    "min_path": min_path_ordering,
}


@dataclass(frozen=True)
class ExperimentRun:
    domain_ordering: List[Domain]
    domain_samples: Dict[Text, DomainSplit]

    def continually_train(self, model: CLModel) -> "ExperimentResults":
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentResults:
    experiment_run: ExperimentRun
    accuracies: List[Dict[Text, float]]
