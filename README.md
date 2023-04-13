# Effect of Domain Ordering while Continually Learning Intent Recognition

## Instructions
The main module is `run.py`. An example execution looks something like this.

<img width="517" alt="image" src="https://user-images.githubusercontent.com/1230386/231894580-18eea635-239b-4635-83fb-0411f8b6005a.png">

## Command Line Arguments

- `--mode`: Mode to run in (type: str).
- `--cl_super_run_label`: Superlabel of the CL run to use (type: str, optional).
- `--cl_lr_schedule`: Learning rate schedule to use (type: str, default: "constant").
- `--lr`: Learning rate to use for training (type: float, default: 1e-4).
- `--num_train_epochs`: Number of epochs to train for (type: int, default: 9).
- `--cl_run_dir`: Directory to store CL runs in (type: str, default: "../cl_runs").
- `--cl_checkpoint_dir`: Directory to store CL checkpoints in (type: str, default: "../cl_checkpoints").
- `--results_dir`: Directory to store results in (type: str, default: "../cl_results").
- `--cl_experience_replay_size`: Number of samples to use for experience replay (type: int, default: 10).
- `--deepspeed_config`: Path to deepspeed config file (type: str, default: "../ds_config.json").
- `--local_rank`: Local rank for distributed training (type: int, default: -1).

- `--num_domains_per_run`: Number of domains to use in a CL run (type: int, default: 5).
- `--num_runs`: Number of CL runs to perform (type: int, default: 5).

- `--ordering_strategy`: Domain ordering to use (type: str, default: "random").

- `--input_max_length`: Maximum length of input sequence (type: int, default: 512).
- `--ctx_window_size`: Number of previous turns in the context (type: int, default: 3).
- `--fp16`: Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit (action: store_true).

- `--limit_n_samples`: Limit the number of total samples per domain (type: int, default: 200).
- `--val_size_per_domain`: Percentage of samples to use for validation (type: float, default: 0.10).
- `--test_size_per_domain`: Percentage of samples to use for testing (type: float, default: 0.25).
- `--train_batch_size`: Batch size for training (type: int, default: 64).
- `--eval_batch_size`: Batch size for evaluation (type: int, default: 64).
- `--shuffle_within_domain`: Whether to shuffle the data within a domain (type: bool, default: False).



