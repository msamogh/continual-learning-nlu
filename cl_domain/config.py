import argparse


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("--mode", type=str, help="Mode to run in")
    args.add_argument(
        "--cl_super_run_label",
        required=False,
        type=str,
        help="Superlabel of the CL run to use",
    )
    args.add_argument(
        "--cl_lr_schedule",
        type=str,
        default="constant",
        help="Learning rate schedule to use",
    )
    args.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate to use for training"
    )
    args.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of epochs to train for"
    )
    args.add_argument(
        "--cl_run_dir",
        type=str,
        default="../cl_runs",
        help="Directory to store CL runs in",
    )
    args.add_argument(
        "--cl_checkpoint_dir",
        type=str,
        default="../cl_checkpoints",
        help="Directory to store CL checkpoints in",
    )
    args.add_argument(
        "--results_dir",
        type=str,
        default="../cl_results",
        help="Directory to store results in",
    )
    args.add_argument(
        "--cl_experience_replay_size",
        type=int,
        default=10,
        help="Number of samples to use for experience replay",
    )

    args.add_argument(
        "--deepspeed_config",
        type=str,
        default="../ds_config.json",
        help="Path to deepspeed config file",
    )
    args.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    add_experiment_args(args)
    add_dataloader_args(args)
    add_tokenizer_args(args)
    add_cl_run_args(args)

    return vars(args.parse_args())


def add_cl_run_args(args):
    args.add_argument(
        "--num_domains_per_run",
        type=int,
        default=5,
        help="Number of domains to use in a CL run",
    )
    args.add_argument(
        "--num_runs", type=int, default=5, help="Number of CL runs to perform"
    )


def add_experiment_args(args):
    args.add_argument(
        "--ordering_strategy", type=str, default="random", help="Domain ordering to use"
    )


def add_tokenizer_args(args):
    args.add_argument(
        "--input_max_length",
        type=int,
        default=512,
        help="Maximum length of input sequence",
    )
    args.add_argument(
        "--ctx_window_size",
        type=int,
        default=3,
        help="Number of previous turns in the context",
    )
    # Bool for fp16
    args.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )


def add_dataloader_args(args):
    args.add_argument(
        "--limit_n_samples",
        type=int,
        default=200,
        help="Limit the number of total samples per domain",
    )
    args.add_argument(
        "--val_size_per_domain",
        type=float,
        default=0.10,
        help="Percentage of samples to use for validation",
    )
    args.add_argument(
        "--test_size_per_domain",
        type=float,
        default=0.25,
        help="Percentage of samples to use for testing",
    )
    args.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size for training"
    )
    args.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    args.add_argument(
        "--shuffle_within_domain",
        type=bool,
        default=False,
        help="Whether to shuffle the data within a domain",
    )
