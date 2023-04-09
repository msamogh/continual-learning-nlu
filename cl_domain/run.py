import pickle
import random
from pathlib import Path
from typing import *

import pickle
import randomname

from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.train import continually_train, get_training_args
from cl_domain.evaluation import evaluate_all_models_over_all_domains
from cl_result import CLRunResult
from domain_ordering import ORDERINGS


def generate_experiment_input(args: Dict[Text, Any], label: Text) -> "CLRunInput":
    # Get all data samples.
    domains: Dict[Text, Domain] = Domain.generate_samples(args["ctx_window_size"])

    # Randomly sample a subset of domains.
    domains = {
        domain_name: domain
        for domain_name, domain in random.sample(list(domains.items()), args["num_domains_per_run"])
    }

    # Split each domain into train, val, and test.
    subsamples = {
        domain_name: DomainSplit.get_fixed_n_split(domain, args["limit_n_samples"], args["test_size_per_domain"], args["val_size_per_domain"])
        for domain_name, domain in domains.items()
    }

    # Order the domains.
    ordering_fn = ORDERINGS[args["ordering_strategy"]]
    ordering = [domains[key] for key in ordering_fn(domains)]

    # Assemble a CLRunInput object and return it.
    from cl_domain.cl_run import CLRunInput
    return CLRunInput(
        label=label,
        domain_ordering=ordering,
        domain_wise_splits=subsamples
    )



def generate_data(args):
    # Generate domain ordering and domain-wise splits.
    # Generate both a pickle and a text file with the ordering.
    super_run_label = randomname.get_name()
    base_run_dir = Path(
        f"../cl_runs/{args['ordering_strategy']}-{super_run_label}")
    if not Path(base_run_dir).exists():
        Path(base_run_dir).mkdir(parents=True)
    # Generate cl_run_inputs for num_runs runs.
    for i in range(args["num_runs"]):
        cl_run_label = f"run_{i}"
        cl_run_input = generate_experiment_input(args, cl_run_label)
        pickle.dump(cl_run_input,
                    open(f"{base_run_dir}/{cl_run_label}.pkl", "wb"))
        with open(f"{base_run_dir}/{cl_run_label}.txt", "w") as f:
            # Write domain names of the ordering in each line.
            f.write("\n".join([domain.domain_name for domain in
                               cl_run_input.domain_ordering]))

    print(f"Finished generating: {super_run_label}")


def train(args):
    # Initialize HuggingFace Trainer
    training_args = get_training_args(args)

    # Read all cl_run_inputs from one particular ordering and
    # continually train them one by one.
    for cl_run_label in Path(f"../cl_runs/{args['cl_super_run_label']}").glob(
            "*.pkl"):
        print(f"Training {args['cl_super_run_label']}/{cl_run_label.stem}...")
        cl_run_input = pickle.load(open(cl_run_label, "rb"))
        continually_train(args, training_args, cl_run_input)

    print("Training finished.")


def evaluate(args):
    # Read all cl_run_inputs from one particular ordering and
    # continually evaluate them one by one.
    print(f"PHASE 2: Evaluating.")
    for cl_run_label in Path(
            f"../cl_runs/{args['cl_super_run_label']}").glob("*.pkl"):
        print(
            f"Evaluating {args['cl_super_run_label']}/{cl_run_label.stem}...")

        cl_run_input = pickle.load(open(cl_run_label, "rb"))
        result_matrix = evaluate_all_models_over_all_domains(args, cl_run_input)
        print(result_matrix)
        cl_run_result = CLRunResult(
            cl_run_label=cl_run_label.stem,
            result_matrix=result_matrix
        )
        print(f"Average forgetting: {cl_run_result.avg_forgetting}")
        print(f"Average accuracy: {cl_run_result.avg_accuracy}")

        # Save the result.
        base_results_dir = Path(f"../cl_results/{args['cl_super_run_label']}")
        if not base_results_dir.exists():
            base_results_dir.mkdir(parents=True)
        pickle.dump(
            cl_run_result,
            (base_results_dir / f"{cl_run_label.stem}.pkl").open("wb")
        )

    print("Evaluation finished.")


if __name__ == "__main__":
    args = get_args()

    if args["mode"] == "generate_data":
        generate_data(args)
    elif args["mode"] == "train":
        train(args)
    elif args["mode"] == "evaluate":
        evaluate(args)
    else:
        raise ValueError(f"Unknown mode {args['mode']}.")
