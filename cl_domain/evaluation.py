from collections import defaultdict

import numpy as np

from cl_domain.domain import *


def avg_forgetting(results: "ExperimentResults") -> float:
    """Calculate the average forgetting.

    Forgetting is defined as the average difference between the best accuracy
    achieved during training and the final accuracy for all tasks (except the
    last task).

    See https://ai.googleblog.com/2022/04/learning-to-prompt-for-continual.html.
    """
    domain_wise_best_accs = defaultdict(float)
    for accuracies in results.cl_run_accuracies[:-1]:
        for domain in results.cl_run_input.domain_ordering:
            domain_wise_best_accs[domain] = max(domain_wise_best_accs[domain],
                                                accuracies[domain.domain_name])
    return np.average([
        results.cl_run_accuracies[-1][domain.domain_name] - domain_wise_best_accs[domain.domain_name]
        for domain in results.cl_run_input.domain_ordering
    ])


def avg_accuracy(results: "ExperimentResults") -> float:
    for accuracies in results.cl_run_accuracies:
        pass
    return 0


if __name__ == "__main__":
    pass
