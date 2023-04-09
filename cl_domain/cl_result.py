from functools import lru_cache
from typing import *
from dataclasses import dataclass

import numpy as np

from cl_run import CLRunInput


@dataclass(frozen=True)
class CLRunResult:
    cl_run_label: Text
    result_matrix: np.ndarray

    @property
    @lru_cache()
    def avg_accuracy(self) -> float:
        return np.mean(self.result_matrix[-1, :])

    @property
    @lru_cache()
    def avg_forgetting(self) -> float:
        return


@dataclass(frozen=True)
class CLSuperRunResult:
    cl_super_run_label: Text
    cl_run_results: List[CLRunResult]

    def write_to_file(self):
        for cl_run_result in self.cl_run_results:
            cl_run_result.write_to_file()
