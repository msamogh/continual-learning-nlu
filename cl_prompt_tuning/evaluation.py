from typing import *
import numpy as np
from collections import defaultdict

from run import *


def forgetting(run: CLRun) -> float:
    """Forgetting is defined as the average difference between the best accuracy
    achieved during training and the final accuracy for all tasks (except the
    last task)."""
    # https://ai.googleblog.com/2022/04/learning-to-prompt-for-continual.html
    task_wise_best_accs = defaultdict(float)
    for accuracies in run.ordered_accuracies[:-1]:
        for task in run.ordered_tasks:
            task_wise_best_accs[task] = max(task_wise_best_accs[task],
                                            accuracies[task])
    return np.average([
        run.ordered_accuracies[-1][task] - task_wise_best_accs[task]
        for task in run.ordered_tasks
    ])


if __name__ == "__main__":
    pass
