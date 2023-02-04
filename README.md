# An Exploration of Rehearsal Methods for NLU in Task-Oriented Dialogue Systems

This repository is based off Andrea Madotto's [Github repo](https://github.com/andreamad8/ToDCL) and the EMNLP paper *
*Continual Learning for Task-Oriented Dialogue Systems** (Accepted in EMNLP
2021) [[PDF]](https://arxiv.org/abs/2012.15504).

## Installation

The Continual Learning benchmark is created by jointly pre-processing four task-oriented dataset such
as [Task-Master (TM19)](https://github.com/google-research-datasets/Taskmaster.git), [Task-Master 2020 (TM20)](https://github.com/google-research-datasets/Taskmaster.git), [Schema Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git)
and [MultiWoZ](https://github.com/budzianowski/multiwoz.git). To download the dataset, and setup basic package use:

```
pip install -r requirements.txt
cd data
bash download.sh
```

If you are interested in the pre-processing, please check ```utils/preprocess.py``` and ```utils/dataloader.py```.

## Basic Running

In this codebase, we implemented several baselines such as MULTI, VANILLA, L2, EWC, AGEM, LAMOL, REPLAY, ADAPTER, and
four ToDs settings such as INTENT, DST, NLG, E2E. An example for running the NLG task with a VANILLA method is:

```
CUDA_VISIBLE_DEVICES=0 python train.py --CL VANILLA --task_type NLG
```

Different CL methods uses different hyperparamters. For example, in REPLAY you can tune the episodic memory size as
following:

```
CUDA_VISIBLE_DEVICES=0 python train.py --CL REPLAY --task_type NLG --episodic_mem_size 10
```

this will randomly sample 10 example per task, and replay it while learning new once. A full example to run the baseline
is for example:

```
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL MULTI 
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL VANILLA --n_epochs 1 
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL L2 --reg 0.01 --n_epochs 1 
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL EWC --reg 0.01 --n_epochs 1
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL AGEM --episodic_mem_size 100 --reg 1.0 --n_epochs 1
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL LAMOL --percentage_LAM0L 200 --n_epochs 1
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL REPLAY --episodic_mem_size 100 --n_epochs 1
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL ADAPTER --bottleneck_size 75 --lr 6.25e-3 --n_epochs 10 --n_epochs 1
```

## Hyperparameters

### Adapter

```
python train.py --task_type E2E --CL ADAPTER --bottleneck_size 300 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --task_type INTENT --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --task_type NLG --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --task_type DST --CL ADAPTER --bottleneck_size 100 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
```

### Replay

```
python train.py --task_type E2E --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --task_type INTENT --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --task_type NLG --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --task_type DST --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
```

## Evaluation

```
python scorer.py --model_checkpoint runs_INTENT/BEST/ --task_type INTENT
python scorer.py --model_checkpoint runs_DST/BEST/ --task_type DST
python scorer.py --model_checkpoint runs_NLG/BEST/ --task_type NLG
python scorer.py --model_checkpoint runs_E2E/BEST/ --task_type E2E
```

### Modularized

| Name    | INTENT    | JGA       | BLEU    | EER       |
|---------|-----------|-----------|---------|-----------|
| VANILLA | 0.0303205 | 0.102345  | 10.3032 | 0.181644  |
| L2      | 0.0346528 | 0.0923626 | 11.0159 | 0.189819  |
| EWC     | 0.0283001 | 0.0998913 | 9.65351 | 0.203158  |
| AGEM    | 0.102224  | 0.0965043 | 4.61297 | 0.360167  |
| LAML    | 0.0262127 | 0.0923302 | 3.49649 | 0.35664   |
| REPLAY  | 0.800088  | 0.394993  | 21.4832 | 0.0559855 |
| ADAPTER | 0.841951  | 0.37381   | 21.7719 | 0.163975  |
| MULTI   | 0.875002  | 0.500357  | 26.1462 | 0.0341823 |

### E2E

| Name    | INTENT    | JGA       | BLEU    | EER      |
|---------|-----------|-----------|---------|----------|
| VANILLA | 0.0264631 | 0.0986375 | 6.45    | 0.499676 |
| L2      | 0.0239718 | 0.069225  | 6.02459 | 0.553715 |
| EWC     | 0.025299  | 0.101422  | 4.72    | 0.572742 |
| AGEM    | 0.303349  | 0.109677  | 4.66216 | 0.651552 |
| LAML    | 0.0269017 | 0.0939656 | 3.55622 | 0.638889 |
| REPLAY  | 0.785325  | 0.297534  | 16.2668 | 0.190309 |
| ADAPTER | 0.906857  | 0.35059   | 16.5768 | 0.331949 |
| MULTI   | 0.954546  | 0.488995  | 23.6073 | 0.12558  |

# Acknowledgement

I would like to thanks [Saujas Vaduguru](saujas.vaduguru@mila.quebec), [Qi Zhu](zhuq96@gmail.com),
and [Maziar Sargordi](maziar.sargordi@mila.quebec) for helping with debugging the code. 
