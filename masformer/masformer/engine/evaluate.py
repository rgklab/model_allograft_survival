import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from joblib import dump

from torch import nn
from lifelines.utils import concordance_index
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from masformer.engine.main import MASFormerModel
from masformer.data.load_data_masformer import load_data_masformer, load_optn_eval_masformer
from masformer.data.load_data import load_SRTR_static_df
from masformer.data.load_OPTN import load_OPTN_dataset
from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.models.transformer import MASFormer, MASFormer_torch
from masformer.data.dataset import SRTR
from utils import dynamic_c_index, dynamic_c_index_mean, dynamic_c_index_avg


def load_evaluate_dataset(feature: str, dataset: str, OPTN = None):

    seed_everything(42)

    if OPTN:
        val, test = load_optn_eval_masformer("graft", feature, OPTN)
    else:
        _, val, test = load_data_masformer("graft", feature)
    features, labels = eval(dataset)
    features[np.isnan(features)]=0
    val_loader = DataLoader(SRTR(features, labels, False), batch_size=128, shuffle=False, 
                            num_workers=10)

    ckpt = f"experiments/checkpoints/masformer_{feature}.ckpt"
    model = MASFormerModel.load_from_checkpoint(checkpoint_path=ckpt)
    trainer = pl.Trainer(gpus=1)

    predictions = trainer.predict(model=model, dataloaders=val_loader)

    harzards = np.concatenate(predictions)
    # surv = np.concatenate(predictions)

    if OPTN:
        graft_val_dyn, graft_test_dyn = _load_optn_eval(OPTN, "dynamic")
        graft_val_sta, graft_test_sta = _load_optn_eval(OPTN, "static")
    else:
        _, graft_val_sta, graft_test_sta = load_SRTR_static_df("graft")
        _, graft_val_dyn, graft_test_dyn = load_dynamic_cox("graft", "ff")
    # if not OPTN:
    #     graft_val_dyn = graft_val_dyn[graft_val_dyn.PX_ID != 470812]


    if dataset == "val":
        data = graft_val_dyn.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    else:
        data = graft_test_dyn.drop(columns=["EVENT"]).merge(graft_test_sta, on="PX_ID", how="left")
    data = data[["PX_ID", "EVENT", "TIME", "start"]]
    data["RISK"] = harzards

    return data


def mean_tdci(feature: str, dataset: str):
    data = load_evaluate_dataset(feature, dataset)

    ci = dynamic_c_index_mean(data, "RISK", "EVENT", "TIME", "start", alpha=0.95, iterations=1000, n_jobs=25)
    
    avg = dynamic_c_index_avg("dynamic", data, "RISK", "EVENT", "TIME", "start")
    print(f"{avg} {ci}")

    out_file = f"experiments/results/model_{dataset}_mean_tdci_Apr_09_2023.txt"

    with open(out_file, "a") as f:
        f.write(f"masformer {feature}: {avg}, {ci} \n")


def optn_mean_tdci(feature: str, OPTN: int, dataset: str = "test"):
    val = load_evaluate_dataset(feature, "val", OPTN)
    test = load_evaluate_dataset(feature, "test", OPTN)

    data = pd.concat([val, test])
    data = data.reset_index(drop=True)
    
    ci = dynamic_c_index_mean(data, "RISK", "EVENT", "TIME", "start", alpha=0.95, iterations=1000, n_jobs=50)
    
    avg = dynamic_c_index_avg("dynamic", data, "RISK", "EVENT", "TIME", "start")
    print(f"{avg} {ci}")

    out_file = f"experiments/results/OPTN_model_{dataset}_mean_tdci.txt"

    with open(out_file, "a") as f:
        f.write(f"OPTN {OPTN} masformer {feature}: {avg}, {ci} \n")


def tdci_matrix(feature: str, dataset: str, ci: bool = False):
    
    alpha = 0.95; NUM_ITERATIONS = 1000
    t_times = [0.5, 1, 3, 5, 7, 9] 
    delta_t_times = [1, 3, 5, 7, 9, float("inf")]

    tt = list(map(lambda x : "t="+str(x), t_times))
    dt = list(map(lambda x : "\u0394t="+str(x), delta_t_times))

    data = load_evaluate_dataset(feature, dataset)

    if ci:
        matrix = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                        "start", t_times, delta_t_times, alpha, NUM_ITERATIONS, ci=True, n_jobs=20)
    else:
        matrix = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                    "start", t_times, delta_t_times, alpha, NUM_ITERATIONS, ci=False, n_jobs=20)
        matrix = matrix.astype(float).round(3)


    matrix.index = tt; matrix.columns = dt
    matrix = matrix.astype(float).round(3)
    print(matrix) 
    dump(matrix,  f"experiments/results/{feature}_masformer_matrix.pkl")



def _load_optn_eval(OPTN: int, mode: str, outcome: str="graft"):
    # load val and test portion of each OPTN region

    # for models trained on all data, when evaluating on OPTN regions,
    # we need to make sure that we are not evaluating its training set,
    # obtain val/test samples from that region.
    _, val, test = load_dynamic_cox(outcome, "ff")
    val_ids, test_ids = val.PX_ID.unique(), test.PX_ID.unique()

    optn_train, optn_val = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="dynamic")
    optn_train_sta, optn_val_sta = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="static")

    optn_data, optn_data_sta = pd.concat([optn_train, optn_val]), pd.concat([optn_train_sta, optn_val_sta])

    graft_val, graft_test = optn_data[optn_data.PX_ID.isin(val_ids)], optn_data[optn_data.PX_ID.isin(test_ids)]
    graft_val_sta = optn_data_sta[optn_data_sta.PX_ID.isin(val_ids)]
    graft_test_sta = optn_data_sta[optn_data_sta.PX_ID.isin(test_ids)]

    if mode == "dynamic":
        return graft_val, graft_test
    else:
        return graft_val_sta, graft_test_sta


if __name__ == "__main__":
    import fire

    fire.Fire()