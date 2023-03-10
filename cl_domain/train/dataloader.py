from typing import *

import torch
import torch.utils.data as data
from datasets import Dataset

from cl_domain.experiment import CLRunInput


def get_dataloader(cl_run_input: CLRunInput, domain_name: Text, split: Text):
    # For domain "domain_name", get the train, val, and test splits.
    samples = cl_run_input.domain_wise_splits[domain_name]
    # TODO convert to training sample


