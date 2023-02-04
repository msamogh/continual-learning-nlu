from dataclasses import dataclass
from typing import *


@dataclass
class Task:
    task_name: Text


@dataclass
class CLRun:
    ordered_tasks: List[Task]
    ordered_accuracies: List[Dict[Task, float]]

    def __len__(self):
        return len(self.ordered_tasks)
