# Equivariant Neural Diffusion for Molecule Generation

This repository contains the implementation accompanying "Equivariant Neural Diffusion for Molecule Generation" (NeurIPS
2024).

## Installation

```
# clone this repo
git clone https://github.com/frcnt/equivariant-neural-diffusion.git

# move to the root directory
cd equivariant-neural-diffusion/

# create an environment
pip install uv
uv venv .venv --python 3.11 
source .venv/bin/activate


uv pip install -e .
```

## Getting started on QM9

### Pre-processing the data

To preprocess `QM9` with the provided splits, the following command can be run.

It should take 2 minutes approximately.

```
export TARGET_DIR="data/qm9"
export SPLIT_FILE="data/qm9/splits.json"

python scripts/preprocess_qm9.py --target_dir $TARGET_DIR --split_file $SPLIT_FILE
```

### Training END

By default, the example config expects the env variables `DATA_PATH` and `LOG_PATH` to be defined.

```
export DATA_PATH="data/" 
export LOG_PATH="path/to/where/to/save/logs-and-checkpoints"

export CONFIG_NAME="train_qm9" # without .yaml extension

python src_end/train.py -cn $CONFIG_NAME 
```

### Evaluating END

A trained model can be evaluated by running a command similar to,

```
export CKPT_PATH="path/to/file.ckpt" 
export INFOS_PATH="data/qm9/preprocessed/train_infos.json"

python src_end/eval.py \
--ckpt_path $CKPT_PATH \
--infos_path $INFOS_PATH \
--n_samples 10_000 \
--n_integration_steps 1000 \
--n_seeds 3
```

In addition to evaluating metrics, the script saves samples in a directory called `eval`
at the root of the `CKPT_PATH`.

## Using END on your own dataset

#### Data

It can be done by pre-processing the data and saving it in `pt` files containing an iterable of `Data` objects. Each of
these `Data` objects should contain at least two fields: `pos` containing the atomic positions, and `h` containing the
atomic features.

#### Metrics

A dataset-specific `Metrics` class can be created to log relevant metrics during validation --- inspiration can be taken
from `QM9Metrics`.

## Citation

If you find this work useful, please consider citing our paper:

```
@article{cornet2024equivariant,
  title={Equivariant neural diffusion for molecule generation},
  author={Cornet, Fran{\c{c}}ois and Bartosh, Grigory and Schmidt, Mikkel and Andersson Naesseth, Christian},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={49429--49460},
  year={2024}
}
```
