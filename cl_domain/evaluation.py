from collections import defaultdict

import numpy as np

from cl_domain.domain import *


from transformers import EvalPrediction


def create_compute_metrics(tokenizer):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        # Find the indices with the highest probabilities along the last axis
        preds = np.argmax(logits[0], axis=-1)

        # Convert token ids back to text labels
        preds_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred
                       in preds]
        labels_texts = [tokenizer.decode(label, skip_special_tokens=False) for
                        label in labels]

        print(list(zip(preds_texts, labels_texts)))

        # Compare predicted labels with ground truth labels
        correct_predictions = np.array(
            [pred == label for pred, label in zip(preds_texts, labels_texts)])

        # Calculate accuracy
        accuracy = np.sum(correct_predictions) / len(labels_texts)

        return {"accuracy": accuracy}

    return compute_metrics


def avg_forgetting(results: "CLRunResult") -> float:
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


def avg_accuracy(results: "CLRunResult") -> float:
    """Calculate the average accuracy.

    In a continual learning setting, this is the average accuracy across all
    domains at the end of training.
    """
    return np.average(results.cl_run_accuracies[-1].values())


if __name__ == "__main__":
    pass
