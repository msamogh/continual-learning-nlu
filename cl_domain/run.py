from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.experiment import CLRunInput
from cl_domain.train import continually_train, init_trainer
from domain_ordering import ORDERINGS


def generate_experiment_input(args):
    domains = Domain.generate_samples(args["ctx_window_size"])
    ordering_fn = ORDERINGS[args["ordering"]]
    ordering = ordering_fn(domains)
    subsamples = {
        domain: DomainSplit.subsample(domain, args["train_size_per_domain"])
        for domain in ordering
    }
    return CLRunInput(
        domain_ordering=ordering,
        domain_wise_samples=subsamples
    )


def run_training_experiment(cl_run_input: CLRunInput):
    pass


if __name__ == "__main__":
    args = get_args()

    cl_run_input = generate_experiment_input(args)
    trainer = init_trainer(cl_run_input)
    continually_train(trainer, cl_run_input)

    print("Done")
