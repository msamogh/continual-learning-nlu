from cl_domain.domain import *

import torch.utils.data as data


@dataclass(frozen=True)
class CLRunInput:
    label: Text
    domain_ordering: List[Domain]
    domain_wise_splits: Dict[Text, DomainSplit]

    def get_ordered_dataloaders(
        self, args
    ) -> Iterable[Tuple[Domain, Dict[Text, data.DataLoader]]]:
        from train import get_dataloader
        from train import TOKENIZER

        domain_wise_dataloaders = {
            domain.domain_name: {
                split: get_dataloader(args, self, domain.domain_name, split, TOKENIZER)
                for split in ["train", "val", "test"]
            }
            for domain in self.domain_ordering
        }
        for domain in self.domain_ordering:
            yield domain, domain_wise_dataloaders[domain.domain_name]
