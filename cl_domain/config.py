import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--ctx_window_size", type=int, default=3, help="Number of previous turns in the context")
    args.add_argument("--train_size_per_domain", type=int, default=100, help="Number of samples to use for training")
    args.add_argument("--ordering", type=str, default="random", help="Ordering to use for the domains")
    return vars(args.parse_args())
