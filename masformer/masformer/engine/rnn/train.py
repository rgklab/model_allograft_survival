import pytorch_lightning as pl
from functools import partial
import argparse
from glob import glob
from joblib import load, dump
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from lifelines.utils import concordance_index
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb

from masformer.data.load_data_masformer import load_data_masformer
from masformer.data.load_data import load_SRTR_static_df
from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.models.rnn import GRU
from masformer.data.dataset import SRTR
from masformer.engine.rnn.main import GRUModel, SRTRDataModule

def run(run_name: str, project_name: str):

    seed_everything(42, workers=True)

    wandb_version = glob(f'/checkpoint/xianggao/rnn/{run_name}/wandb_version*')
    version, resume = None, None

    if len(wandb_version) > 0:
        version_path = wandb_version[0]
        version = load(version_path)
        resume = True
        wandb_logger = WandbLogger(project=project_name, name=str(run_name), id=version, resume=resume)
        wandb.define_metric("tdci_by_harzard", summary="max")
        wandb.define_metric("tdci_by_surv", summary="max")
        ckpt = glob(f'/checkpoint/xianggao/rnn/{run_name}/*last*.ckpt')[0]
        hparams = dict(wandb.config)
        rnn_model = GRUModel(hparams=hparams)

        trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0,
                         max_epochs=80, auto_scale_batch_size=True,
                         callbacks=[
                            #  # save the best model
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_surv', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_surv:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{run_name}')
                         ],
                         logger=wandb_logger
                         )
    
        trainer.fit(rnn_model, SRTRDataModule(feature=hparams["feature"], batch_size=hparams["batch_size"]), ckpt_path=ckpt)
    else:
        wandb.init()
        wandb.define_metric("tdci_by_harzard", summary="max")
        wandb.define_metric("tdci_by_surv", summary="max")
        hparams = dict(wandb.config)
        wandb_logger = WandbLogger(project=project_name, name=str(run_name))
        dump(wandb_logger.version, f"/checkpoint/xianggao/rnn/{run_name}/wandb_version.txt")
        rnn_model = GRUModel(hparams=hparams)
        trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0,
                         max_epochs=80, auto_scale_batch_size=True,
                         callbacks=[
                            #  # save the best model
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_surv', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_surv:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{run_name}')
                         ],
                         logger=wandb_logger
                         )
        trainer.fit(rnn_model, SRTRDataModule(feature=hparams["feature"], batch_size=hparams["batch_size"]))


    # hparams = {"num_heads":4, "d_model":512, "drop_prob":0.1, "num_layers":4, "d_ff":2048,
    #         "learning_rate":1e-4, "coeff":0.1, "coeff2":0.1, "num_features":263, "feature":feature, "batch_size":16}
    # if feature == "mas":
    #     hparams['num_features'] = 6


def test():
    from glob import glob
    from joblib import load, dump

    wandb_version = glob(f'experiments/checkpoints/full_100_epochs/wandb_version*')
    version, resume = None, None

    if len(wandb_version) > 0:
        version_path = wandb_version[0]
        version = load(version_path)
        resume = True

    # wandb_logger = WandbLogger(project="masformer", name="full_100_epochs", id=version, resume=resume)
    wandb_logger = WandbLogger(project="gru", name="full_100_epochs")

    import pdb; pdb.set_trace()


def prepare_args():
    parser = argparse.ArgumentParser(description='specify run name')
    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--run_name', type=str)
    args = parser.parse_args()
    return args


def run_agent(sweep_id: str, run_name: str, project_name: str):

    version_path = f'/checkpoint/xianggao/rnn/{run_name}/wandb_version*'
    wandb_version = glob(version_path)

    train_func = partial(run, run_name=run_name, project_name=project_name)

    if len(wandb_version) > 0:
        train_func()
    else:
        # initial start call to agent
        wandb.agent(sweep_id,
                    count=1,
                    project= project_name,
                    function=train_func)


def final(out_dir: str, feature: str="full"):
    from glob import glob
    from joblib import load, dump

    seed_everything(42, workers=True)

    wandb_version = glob(f'/checkpoint/xianggao/rnn/{out_dir}/wandb_version*')
    version, resume = None, None

    if len(wandb_version) > 0:
        version_path = wandb_version[0]
        version = load(version_path)
        resume = True

    wandb_logger = WandbLogger(project="gru", name=out_dir, id=version, resume=resume)

    dump(wandb_logger.version, f"/checkpoint/xianggao/rnn/{out_dir}/wandb_version.txt")

    trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0,
                         max_epochs=15, auto_scale_batch_size=True,
                         callbacks=[
                            #  # save the best model
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_harzard', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_harzard:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{out_dir}'),
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_surv', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_surv:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{out_dir}')
                         ],
                         logger=wandb_logger
                         )
    # 2 epochs by harzard 0.8385
    hparams = {"d_model":128, "drop_prob":0.5414929955362152, "num_layers":3,
            "learning_rate":1e-3, "coeff":0.01, "num_features":287, "feature":feature, "batch_size":16}
    if feature == "mas":
        # 11 epochs by surv 0.8406
        hparams = {"d_model":256, "drop_prob":0.4879877457248929, "num_layers":1,
            "learning_rate":5e-4, "coeff":0.01, "num_features":6, "feature":feature, "batch_size":8}
    rnn_model = GRUModel(hparams=hparams)

    ckpts = glob(f'/checkpoint/xianggao/rnn/{out_dir}/*last*.ckpt')
    ckpt = None
    if len(ckpts) > 0:
        ckpt = ckpts[0]

    trainer.fit(rnn_model, SRTRDataModule(feature=hparams["feature"], batch_size=hparams["batch_size"]), ckpt_path=ckpt)



if __name__ == "__main__":
    import fire

    fire.Fire()
