import argparse


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("--mode", type=str, help="Mode to run in")
    args.add_argument("--cl_run_label", required=False, type=str, help="Label of the CL run to use")

    add_experiment_args(args)
    add_dataloader_args(args)
    add_tokenizer_args(args)
    add_cl_run_args(args)

    return vars(args.parse_args())


def add_cl_run_args(args):
    args.add_argument("--num_domains_per_run", type=int, default=5, help="Number of domains to use in a CL run")
    args.add_argument("--num_runs", type=int, default=5, help="Number of CL runs to perform")

def add_experiment_args(args):
    args.add_argument("--ordering", type=str, default="random", help="Domain ordering to use")


def add_tokenizer_args(args):
    args.add_argument("--input_max_length", type=int, default=512,
                      help="Maximum length of input sequence")
    args.add_argument("--ctx_window_size", type=int, default=3,
                      help="Number of previous turns in the context")


def add_dataloader_args(args):
    args.add_argument("--train_samples_per_domain", type=int, default=100,
                      help="Number of samples to use for training")
    args.add_argument("--val_size_per_domain", type=float, default=0.15,
                      help="Percentage of samples to use for validation")
    args.add_argument("--test_size_per_domain", type=float, default=0.25,
                      help="Percentage of samples to use for testing")
    args.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    args.add_argument("--shuffle_within_domain", type=bool, default=False, help="Whether to shuffle the data within a domain")
