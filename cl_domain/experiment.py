from cl_domain.domain import *


@dataclass(frozen=True)
class CLRunInput:
    domain_ordering: List[Domain]
    domain_wise_samples: Dict[Text, DomainSplit]


@dataclass(frozen=True)
class CLRunResult:
    cl_run_input: CLRunInput
    cl_run_accuracies: List[Dict[Text, float]]
