from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.experiment import CLRunInput
from cl_domain.train import continually_train, init_trainer
from domain_ordering import ORDERINGS


def generate_experiment_input(args):
    # Get all data samples.
    domains = Domain.generate_samples(args["ctx_window_size"])
    # Split each domain into train, val, and test.
    subsamples = {
        domain: DomainSplit.get_fixed_n_split(domain, args["train_samples_per_domain"], args["test_size_per_domain"], args["val_size_per_domain"])
        for domain in domains
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
    trainer = init_trainer()

    # Calling .fit() on the trainer will train the model on the training set.
    continually_train(trainer, cl_run_input)

    print("Done")
