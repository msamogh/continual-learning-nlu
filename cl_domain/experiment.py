from cl_domain.domain import *

import torch.utils.data as data

from cl_domain.train import get_dataloader, TOKENIZER


@dataclass(frozen=True)
class CLRunInput:
    domain_ordering: List[Domain]
    domain_wise_splits: Dict[Text, DomainSplit]

    def get_ordered_dataloaders(self, args) -> Iterable[Tuple[Domain, Dict[Text, data.DataLoader]]]:
        domain_wise_dataloaders = {
            domain.domain_name: {
                split: get_dataloader(args, self, domain.domain_name, split, TOKENIZER)
                for split in ["train", "val", "test"]
            }
            for domain in self.domain_ordering
        }
        for domain in self.domain_ordering:
            yield domain, domain_wise_dataloaders[domain.domain_name]


@dataclass(frozen=True)
class CLRunResult:
    cl_run_input: CLRunInput
    cl_run_accuracies: List[Dict[Text, float]]
