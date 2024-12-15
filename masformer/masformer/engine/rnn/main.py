import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from lifelines.utils import concordance_index
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from masformer.data.load_data_masformer import load_data_masformer
from masformer.data.load_data import load_SRTR_static_df
from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.models.rnn import GRU
from masformer.data.dataset import SRTR
from utils import dynamic_c_index_avg

class GRUModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hp = hparams
        self.save_hyperparameters()
        self.model = GRU(self.hp["num_features"], self.hp["d_model"], self.hp["num_layers"], 
                        self.hp["drop_prob"])

    def forward(self, x):
        return self.model(x)

    # def forward(self, x, mask=None):
    #     # calculate cumulative harzard (for SHAP)
    #     # where x is concatation of feature+mask
    #     new_sigmoid = self.model(x, mask) * mask
    #     harzard = 1 - new_sigmoid
    #     return torch.sum(harzard, dim=1).unsqueeze(dim=1)

    def training_step(self, batch, batch_idx):
        features, true_durations, mask, label, is_observed = batch
        is_observed_a = is_observed[0]
        is_observed_b = is_observed[1]
        mask_a = mask[0]
        mask_b = mask[1]
        label_a = label[0]
        true_durations_a = true_durations[0]
        true_durations_b = true_durations[1]

        sigmoid_a = self.forward(features[0]) * mask_a
        surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
        loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)

        sigmoid_b = self.forward(features[1]) * mask_b
        surv_probs_b = torch.cumprod(sigmoid_b, dim=1)

        cond_a = (is_observed_a & (true_durations_a < true_durations_b)) & (is_observed_b)
        disc_loss = 0.
        if torch.sum(cond_a) > 0:
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
            mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
            diff = mean_lifetimes_a[cond_a] - mean_lifetimes_b[cond_a]
            true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
            disc_loss += self.hp["coeff"] * torch.mean(nn.ReLU()(true_diff + diff))

        cond_a2 = is_observed_a
        if torch.sum(cond_a2) > 0:
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
            disc_loss += self.hp["coeff"] * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2])
        
        loss += disc_loss

        # self.log('disc_loss', disc_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    # def training_epoch_end(self, outputs):
    #     avg_train_loss = torch.tensor([x['loss'] for x in outputs]).mean()
    #     self.log('avg_train_loss', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        features, true_durations, mask, label, is_observed = batch
        is_observed_a = is_observed[0]
        is_observed_b = is_observed[1]
        mask_a = mask[0]
        mask_b = mask[1]
        label_a = label[0]
        true_durations_a = true_durations[0]
        true_durations_b = true_durations[1]

        sigmoid_a = self.forward(features[0]) * mask_a
        surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
        loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)

        sigmoid_b = self.forward(features[1]) * mask_b
        surv_probs_b = torch.cumprod(sigmoid_b, dim=1)

        cond_a = (is_observed_a & (true_durations_a < true_durations_b)) & (is_observed_b)
        if torch.sum(cond_a) > 0:
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
            mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
            diff = mean_lifetimes_a[cond_a] - mean_lifetimes_b[cond_a]
            true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
            loss += self.hp["coeff"] * torch.mean(nn.ReLU()(true_diff + diff))

        cond_a2 = is_observed_a
        if torch.sum(cond_a2) > 0:
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
            loss += self.hp["coeff"] * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2])

        self.log('val_loss', loss)

        new_mask = torch.where((mask_a == 1.), 1., torch.nan)
        new_sigmoid = self.forward(features[0]) * new_mask
        harzard = 1 - new_sigmoid
        harzard = harzard[~torch.isnan(harzard)]
        new_surv = torch.cumprod(new_sigmoid, dim=1)
        new_surv = new_surv[~torch.isnan(new_surv)]

        return {"harzard": harzard.cpu().numpy(), "surv": new_surv.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        _, graft_val_sta, _ = load_SRTR_static_df("graft")
        _, graft_val_dyn, _ = load_dynamic_cox("graft", "ff")
        graft_val_dyn = graft_val_dyn[graft_val_dyn.PX_ID != 470812]
        harzard = np.concatenate([x['harzard'] for x in outputs])
        surv = np.concatenate([x['surv'] for x in outputs])

        dcox_graft_df = graft_val_dyn.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
        dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
        dcox_graft_df["RISK"] = -surv
        tdci_by_surv = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
        dcox_graft_df["RISK"] = harzard
        tdci_by_harzard = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
        self.log('tdci_by_harzard', tdci_by_harzard)
        self.log('tdci_by_surv', tdci_by_surv)

        # c_index_by_harzard = concordance_index(graft_val_dyn["stop"], -harzard, graft_val_dyn["EVENT"])
        # c_index_by_surv = concordance_index(graft_val_dyn["stop"], surv, graft_val_dyn["EVENT"])
        # self.log('c_index_by_harzard', c_index_by_harzard)
        # self.log('c_index_by_surv', c_index_by_surv)

    def predict_step(self, batch, batch_idx):
        features, true_durations, mask, label, is_observed = batch
        mask_a = mask[0]
        new_mask = torch.where((mask_a == 1.), 1., torch.nan)
        new_sigmoid = self.forward(features[0]) * new_mask
        harzard = 1 - new_sigmoid
        harzard = harzard[~torch.isnan(harzard)]
        new_surv = torch.cumprod(new_sigmoid, dim=1)
        new_surv = new_surv[~torch.isnan(new_surv)]

        return harzard.cpu().numpy()
        # return -new_surv.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp["learning_rate"])
        return optimizer


class SRTRDataModule(pl.LightningDataModule):
    def __init__(self, outcome="graft", feature="full", batch_size=16, num_worker=8) -> None:
        super().__init__()
        train, val, test = load_data_masformer(outcome, feature)
        features, labels = train
        features[np.isnan(features)]=0
        self.train = DataLoader(SRTR(features, labels, True), batch_size=batch_size, shuffle=True, 
                            num_workers=num_worker)
        features, labels = val
        features[np.isnan(features)]=0
        self.val = DataLoader(SRTR(features, labels, False), batch_size=batch_size, shuffle=False, 
                            num_workers=num_worker)
        features, labels = test
        features[np.isnan(features)]=0
        self.test = DataLoader(SRTR(features, labels, False), batch_size=batch_size, shuffle=False, 
                            num_workers=num_worker)

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test


def run(out_dir: str, feature: str="full"):
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
                         max_epochs=100, auto_scale_batch_size=True,
                         callbacks=[
                            #  # save the best model
                            # ModelCheckpoint(save_top_k=1, monitor='c_index_by_harzard', mode='max',
                            #                             filename="{epoch}-{step}-{c_index_by_harzard:.3f}",
                            #                             save_last=True,
                            #                             dirpath=f'/checkpoint/xianggao/masformer/{out_dir}'),
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_harzard', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_harzard:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{out_dir}'),
                            # ModelCheckpoint(save_top_k=1, monitor='c_index_by_surv', mode='max',
                            #                             filename="{epoch}-{step}-{c_index_by_surv:.3f}",
                            #                             save_last=True,
                            #                             dirpath=f'/checkpoint/xianggao/masformer/{out_dir}'),
                            ModelCheckpoint(save_top_k=1, monitor='tdci_by_surv', mode='max',
                                                        filename="{epoch}-{step}-{tdci_by_surv:.3f}",
                                                        save_last=True,
                                                        dirpath=f'/checkpoint/xianggao/rnn/{out_dir}')
                         ],
                         logger=wandb_logger
                         )
    hparams = {"num_heads":4, "d_model":512, "drop_prob":0.1, "num_layers":4, "d_ff":2048,
            "learning_rate":1e-4, "coeff":0.1, "num_features":287, "feature":feature, "batch_size":128}
    if feature == "mas":
        hparams['num_features'] = 6
    masformer_model = GRUModel(hparams=hparams)

    ckpts = glob(f'/checkpoint/xianggao/rnn/{out_dir}/*last*.ckpt')
    ckpt = None
    if len(ckpts) > 0:
        ckpt = ckpts[0]

    trainer.fit(masformer_model, SRTRDataModule(feature=hparams["feature"], batch_size=hparams["batch_size"]), ckpt_path=ckpt)


def debug(feature:str="full"):
    print("Running Debug Mode")
    seed_everything(42, workers=True)
    hparams = {"d_model":512, "drop_prob":0.1, "num_layers":2,
            "learning_rate":1e-4, "coeff":0.01, "num_features":287, "feature":feature, "batch_size":16}
    trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    if feature == "mas":
        hparams['num_features'] = 6

    masformer_model = GRUModel(hparams=hparams)
    trainer.fit(masformer_model, SRTRDataModule(feature=hparams["feature"], batch_size=hparams["batch_size"]))


if __name__ == "__main__":
    import fire

    fire.Fire()
