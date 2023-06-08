#!/bin/sh
#SBATCH --account=boyer
#SBATCH --qos=boyer
#SBATCH --partition=gpu
#SBATCH --gpus=a100:7
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 2    # cores requested
#SBATCH --mem=60gb  # memory in Mb
#SBATCH -o out_cl  # send stdout to outfile
#SBATCH -e err_cl  # send stderr to errfile
#SBATCH -t 10:00:00  # time requested in hour:minute:second

# Load necessary modules
module load git
module load python/3.8
module load cuda/11.4.3

# Remove any existing old files
# rm -rf continual-learning-nlu
# rm out_cl
# rm err_cl
# rm venv

# Clone the GitHub repository
#git clone https://github.com/msamogh/continual-learning-nlu.git
cd continual-learning-nlu
# git pull

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Create and activate a virtual environment
pip install -r requirements.txt

# Change directory
cd cl_domain/

# Random Generate dataset + Train + Evaluate
# echo "Strategy = Random"
# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode train \
#     --cl_super_run_label random-flat-tee \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy random \
#     --num_train_epochs 18 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \

# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode evaluate \
#     --cl_super_run_label random-flat-tee \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy random \
#     --num_train_epochs 5 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \

# Max path Generate dataset + Train + Evaluate
# echo "Strategy = Max Path"
# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode train \
#     --cl_super_run_label max_path-vicious-tree \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy max_path \
#     --num_train_epochs 18 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \

# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode evaluate \
#     --cl_super_run_label max_path-vicious-tree \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy max_path \
#     --num_train_epochs 5 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \


# Min path Generate dataset + Train + Evaluate
echo "Strategy = Min Path"

# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode generate_data \
#     --cl_super_run_label tahoe-word \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy min_path \
#     --num_train_epochs 18 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \

PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
    --mode train \
    --cl_super_run_label max_path-studio-vest \
    --cl_checkpoint_dir ../cl_checkpoints \
    --cl_run_dir ../cl_runs \
    --results_dir ../cl_results \
    --ordering_strategy max_path \
    --num_train_epochs 18 \
    --num_runs 22 \
    --num_domains_per_run 5 \
    --cl_lr_schedule constant \
    --limit_n_samples 180 \
    --val_size_per_domain 0.01 \
    --cl_experience_replay_size 20 \
    --eval_batch_size 4 \
    --train_batch_size 4 \

PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
    --mode evaluate \
    --cl_super_run_label max_path-studio-vest \
    --cl_checkpoint_dir ../cl_checkpoints \
    --cl_run_dir ../cl_runs \
    --results_dir ../cl_results \
    --ordering_strategy max_path \
    --num_train_epochs 5 \
    --num_runs 22 \
    --num_domains_per_run 5 \
    --cl_lr_schedule constant \
    --limit_n_samples 180 \
    --val_size_per_domain 0.01 \
    --cl_experience_replay_size 20 \
    --eval_batch_size 4 \
    --train_batch_size 4 \



# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode train \
#     --cl_super_run_label random-plastic-cover \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy random \
#     --num_train_epochs 18 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 4 \
#     --train_batch_size 4 \

# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode evaluate \
#     --cl_super_run_label random-plastic-cover \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy random \
#     --num_train_epochs 5 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule constant \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 4 \
#     --train_batch_size 4 \

# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode all \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy max_path \
#     --num_train_epochs 5 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule linear \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \

# # Generate dataset + Train + Evaluate
# echo "Strategy = Min Path"
# PYTHONPATH=$PYTHONPATH:$(pwd)/.. python run.py \
#     --mode all \
#     --cl_checkpoint_dir ../cl_checkpoints \
#     --cl_run_dir ../cl_runs \
#     --results_dir ../cl_results \
#     --ordering_strategy min_path \
#     --num_train_epochs 5 \
#     --num_runs 22 \
#     --num_domains_per_run 5 \
#     --cl_lr_schedule linear \
#     --limit_n_samples 180 \
#     --val_size_per_domain 0.01 \
#     --cl_experience_replay_size 20 \
#     --eval_batch_size 16 \
#     --train_batch_size 16 \
