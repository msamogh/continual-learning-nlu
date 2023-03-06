from typing import *

import torch
import torch.utils.data as data

from cl_domain.experiment import CLRunInput


def get_dataloader(cl_run_input: CLRunInput, domain_name: Text, split: Text):
    samples = cl_run_input.domain_wise_samples[domain_name]
    # TODO convert to sample

