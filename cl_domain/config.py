import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--ctx_window_size", type=int, default=3, help="Number of previous turns in the context")
    args.add_argument("--train_samples_per_domain", type=int, default=100, help="Number of samples to use for training")
    args.add_argument("--val_size_per_domain", type=float, default=0.15,
                      help="Percentage of samples to use for validation")
    args.add_argument("--test_size_per_domain", type=float, default=0.25,
                      help="Percentage of samples to use for testing")
    args.add_argument("--ordering", type=str, default="random", help="Ordering to use for the domains")
    return vars(args.parse_args())
