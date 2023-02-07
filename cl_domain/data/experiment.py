from typing import *

from cl_domain.data import *
from cl_domain.model import CLModel


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
