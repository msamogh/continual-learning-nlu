from typing import *

import torch
import torch.utils.data as data

from cl_domain.experiment import CLRunInput


def get_dataloader(cl_run_input: CLRunInput, split: Text):
    for domain in cl_run_input.domain_ordering:
        pass

