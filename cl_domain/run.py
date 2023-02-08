from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.experiment import ORDERINGS, ExperimentRun


def generate_experiment_run(args):
    domains = Domain.get_all_domains(args["ctx_window_size"])
    ordering_fn = ORDERINGS[args["ordering"]]
    ordering = ordering_fn(domains)
    subsamples = [
        DomainSplit.subsample(domain, args["train_size_per_domain"])
        for domain in ordering
    ]
    experiment_run = ExperimentRun(
        domain_ordering=ordering,
        domain_samples=subsamples
    )
    return experiment_run


if __name__ == "__main__":
    args = get_args()
    experiment_run = generate_experiment_run(args)
    print("Done")

