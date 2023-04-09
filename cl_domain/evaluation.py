from collections import defaultdict

import numpy as np

from cl_domain.domain import *


from transformers import EvalPrediction, T5ForConditionalGeneration, Trainer


def evaluate_all_models_over_all_domains(args: Dict[Text, Any], cl_run_input: "CLRunInput") -> np.ndarray:
    result_matrix = np.zeros(
        (len(cl_run_input.domain_ordering), len(cl_run_input.domain_ordering))
    )
    for i in range(len(cl_run_input.domain_ordering)):
        model_i = T5ForConditionalGeneration.from_pretrained(
            f"../cl_checkpoints/{args['cl_super_run_label']}/{cl_run_input.label}/after_{i}"
        )
        for j, (domain, domain_wise_dataloader) in enumerate(
                cl_run_input.get_ordered_dataloaders(args)
        ):
            print(f"Evaluating domain {j} with model {i}")
            dl_j = domain_wise_dataloader["test"]
            from train import TOKENIZER
            result_matrix[i, j] = Trainer(
                model=model_i,
                eval_dataset=dl_j,
                compute_metrics=create_compute_metrics(TOKENIZER)
            ).evaluate()["eval_accuracy"]
    return result_matrix



def create_compute_metrics(tokenizer):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        # Find the indices with the highest probabilities along the last axis
        preds = np.argmax(logits[0], axis=-1)

        # Convert token ids back to text labels
        preds_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred
                       in preds]
        labels_texts = [tokenizer.decode(label, skip_special_tokens=True) for
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
