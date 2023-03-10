from cl_domain.domain import *

import torch.utils.data as data


@dataclass(frozen=True)
class CLRunInput:
    domain_ordering: List[Domain]
    domain_wise_splits: Dict[Text, DomainSplit]

    def get_ordered_dataloaders(self) -> Iterable[Tuple[Domain, Dict[Text, data.DataLoader]]]:
        domain_wise_dataloaders = {
            domain_name: {
                split: get_dataloader(self, domain_name, split)
                for split in ["train", "val", "test"]
            }
            for domain_name in self.domain_ordering
        }
        for domain in self.domain_ordering:
            yield domain, domain_wise_dataloaders[domain.name]


@dataclass(frozen=True)
class CLRunResult:
    cl_run_input: CLRunInput
    cl_run_accuracies: List[Dict[Text, float]]
