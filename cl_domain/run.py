from typing import *

from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.experiment import CLRunInput
from cl_domain.train import continually_train, get_training_args
from domain_ordering import ORDERINGS


def generate_experiment_input(args):
    # Get all data samples.
    domains: Dict[Text, Domain] = Domain.generate_samples(args["ctx_window_size"])
    # Split each domain into train, val, and test.
    subsamples = {
        domain_name: DomainSplit.get_fixed_n_split(domain, args["train_samples_per_domain"], args["test_size_per_domain"], args["val_size_per_domain"])
        for domain_name, domain in domains.items()
    }

    # Order the domains.
    ordering_fn = ORDERINGS[args["ordering"]]
    ordering = ordering_fn(domains)

    # Assemble a CLRunInput object and return it.
    return CLRunInput(
        domain_ordering=ordering,
        domain_wise_splits=subsamples
    )


if __name__ == "__main__":
    args = get_args()

    # Generate domain ordering and domain-wise splits.
    cl_run_input = generate_experiment_input(args)

    # Initialize HuggingFace Trainer
    training_args = get_training_args(args)

    # Calling .fit() on the trainer will train the model on the training set.
    continually_train(args, training_args, cl_run_input)

    print("Done")
