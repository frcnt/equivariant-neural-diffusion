import json
import os

import fire as fire
import hydra
import numpy as np
import pytorch_lightning as pl
import torch.cuda
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src_end import utils
from src_end.data.dataset import SampleDataset
from src_end.data.transforms import FullyConnected
from src_end.data.utils import save_json, read_json, save_images
from src_end.lit.module import LitEND
from src_end.utils.utils import get_next_version, load_cfg

log = utils.get_pylogger(__name__)

SEED_DIRNAME = "seed={}"
METRICS_FNAME = "metrics.json"


def build_sample_dataset(infos_path: str, n_samples: int):
    dataset_infos = read_json(json_path=infos_path)
    empirical_distribution = np.array(dataset_infos.get("num_atoms_hist"))

    sample_dataset = SampleDataset(empirical_distribution=empirical_distribution,
                                   n_samples=n_samples,
                                   transform=FullyConnected(key="edge_index"))

    loader = DataLoader(dataset=sample_dataset,
                        batch_size=256,
                        num_workers=1,
                        pin_memory=True)

    return sample_dataset, loader


def load_dm(cfg: DictConfig, n_samples: int):
    print(f"Instantiating datamodule <{cfg.datamodule.get('_target_')}>")
    cfg.datamodule.num_val_subset = n_samples
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    return datamodule


def evaluate(ckpt_path: str,
             infos_path: str,
             n_samples: int = 512,
             n_integration_steps: int = 100,
             n_seeds: int = 3,
             use_ema: bool = True, ):
    job_dir, cfg = load_cfg(checkpoint=ckpt_path)

    eval_root_dir = os.path.join(job_dir, "eval")
    v = get_next_version(eval_root_dir)
    eval_dir = os.path.join(eval_root_dir, f"version_{v}")
    os.makedirs(eval_dir, exist_ok=True)

    json_dict = dict(ckpt_path=ckpt_path,
                     n_integration_steps=n_integration_steps,
                     n_samples=n_samples,
                     n_seeds=n_seeds)

    save_json(json_dict=json_dict, json_path=os.path.join(eval_dir, "cmd_args.json"))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE:", device)
    lit_module: LitEND = hydra.utils.instantiate(cfg.lit_module)
    ckpt = torch.load(ckpt_path, map_location=device)

    (missing_keys, unexpected_keys) = lit_module.load_state_dict(ckpt["state_dict"], strict=False)

    if len(missing_keys) > 0:
        log.warning(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        log.warning(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    lit_module.to(device)
    print("lit_module.device:", lit_module.device)

    sample_dataset, loader = build_sample_dataset(infos_path=infos_path, n_samples=n_samples)
    print(f"> Creating sample dataset with {len(sample_dataset)} entries.")

    lit_module.hparams.n_integration_steps = n_integration_steps
    print(f"> Overriding sampling parameters with provided ones. Integration steps: {n_integration_steps}.")

    seeds = range(n_seeds)

    metrics = None

    for seed in seeds:
        print(f"Starting evaluation for seed: {seed}")
        pl.seed_everything(seed, workers=True)
        save_dir = os.path.join(eval_dir, SEED_DIRNAME.format(seed))
        os.makedirs(save_dir, exist_ok=False)

        atoms_seed = []

        for batch in iter(loader):
            atoms = lit_module.sample(batch.to(device), ema=use_ema)
            lit_module.metrics.update(atoms)
            atoms_seed.extend(atoms)

        # metrics
        m = lit_module.metrics.summarize()
        lit_module.metrics.reset()

        if metrics is None:
            metrics = {k: [m[k]] for k in m}
        else:
            for k in m:
                metrics[k].append(m[k])
        # samples
        save_images(atoms_seed, os.path.join(save_dir, "samples.xyz"))

    # Summary
    summary = {}
    for k in metrics:
        summary[f"{k}"] = {"mean": np.mean(metrics[k]), "std": np.std(metrics[k])}

    summary_path = os.path.join(eval_dir, "summary_metrics.json")
    save_json(summary, json_path=summary_path)

    print(f"Summary saved in '{summary_path}'.")

    print("---------------")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    fire.Fire(evaluate)
