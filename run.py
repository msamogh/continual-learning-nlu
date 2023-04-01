from typing import *

from cl_domain.config import get_args
from cl_domain.domain import Domain, DomainSplit
from cl_domain.experiment import CLRunInput
from cl_domain.train import continually_train, get_training_args
from cl_domain.domain_ordering import ORDERINGS
from nltk.tokenize import sent_tokenize



def generate_experiment_input(args):
    # Get all data samples.
    domains: Dict[Text, Domain] = Domain.generate_samples(args["ctx_window_size"])
    # Split each domain into train, val, and test.
    subsamples = {
        domain_name: DomainSplit.get_fixed_n_split(domain, args["train_samples_per_domain"], args["test_size_per_domain"], args["val_size_per_domain"])
        for domain_name, domain in domains.items()
    }
    domain_wise_clustered_txt = {}
    for key in subsamples.keys():
        sentence = ''
        for test_sample in subsamples[key].test_samples:
            for context in test_sample.context:
                sentence += ' '
                sentence += context.utterance
        domain_wise_clustered_txt[key] = sent_tokenize(sentence)

    # Order the domains.
    #ordering_fn = ORDERINGS[args["ordering"]]
    ordering_fn = ORDERINGS['min_path']
    ordering_key = ordering_fn(domain_wise_clustered_txt)
    ordering = [domains[key] for key in ordering_key]

    # Assemble a CLRunInput object and return it.s
    return CLRunInput(
        domain_ordering=ordering,
        domain_wise_splits=subsamples
    )


if __name__ == "__main__":
    args = get_args()

    # Generate domain ordering and domain-wise splits.
    cl_run_input = generate_experiment_input(args)

    # Initialize HuggingFace Trainer
    #training_args = get_training_args(args)

    # Calling .fit() on the trainer will train the model on the training set.
    #continually_train(args, training_args, cl_run_input)

    print("Done")
