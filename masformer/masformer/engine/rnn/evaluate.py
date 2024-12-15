import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from joblib import dump

from torch import nn
from lifelines.utils import concordance_index
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from masformer.engine.rnn.main import GRUModel
from masformer.data.load_data_masformer import load_data_masformer
from masformer.data.load_data import load_SRTR_static_df
from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.models.rnn import GRU
from masformer.data.dataset import SRTR
from utils import dynamic_c_index, dynamic_c_index_mean, dynamic_c_index_avg


def load_evaluate_dataset(feature: str, dataset: str):
    seed_everything(42)
    _, val, test = load_data_masformer("graft", feature)
    features, labels = eval(dataset)
    features[np.isnan(features)]=0
    val_loader = DataLoader(SRTR(features, labels, False), batch_size=128, shuffle=False, 
                            num_workers=10)

    ckpt = f"experiments/checkpoints/rnn_{feature}.ckpt"
    model = GRUModel.load_from_checkpoint(checkpoint_path=ckpt)
    trainer = pl.Trainer(gpus=1)

    predictions = trainer.predict(model=model, dataloaders=val_loader)

    harzards = np.concatenate(predictions)
    # surv = np.concatenate(predictions)

    _, graft_val_sta, graft_test_sta = load_SRTR_static_df("graft")
    _, graft_val_dyn, graft_test_dyn = load_dynamic_cox("graft", "ff")
    # graft_test_dyn = graft_test_dyn[graft_test_dyn.PX_ID != 330357]
    graft_val_dyn = graft_val_dyn[graft_val_dyn.PX_ID != 470812]

    if dataset == "val":
        data = graft_val_dyn.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    else:
        data = graft_test_dyn.drop(columns=["EVENT"]).merge(graft_test_sta, on="PX_ID", how="left")
    data = data[["PX_ID", "EVENT", "TIME", "start"]]
    data["RISK"] = harzards

    return data

def mean_tdci(feature: str, dataset: str):
    alpha = 0.95
    NUM_ITERATIONS = 1 # TODO - change to 1000

    t_times = [0.5, 1, 3, 5, 7, 9]                     # Reference times
    delta_t_times = [1, 3, 5, 7, 9, float("inf")]    # Prediction horizon times

    tt = list(map(lambda x : "t="+str(x), t_times))
    dt = list(map(lambda x : "\u0394t="+str(x), delta_t_times))

    data = load_evaluate_dataset(feature, dataset)

    masformer_graft_matrix = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                        "start", t_times, delta_t_times, alpha, NUM_ITERATIONS)

    masformer_graft_matrix.index = tt; masformer_graft_matrix.columns = dt

    print(masformer_graft_matrix.astype(float).round(3))

    dump(masformer_graft_matrix.astype(float).round(3), f"experiments/results/{feature}_masformer_matrix.pkl")

    ci = dynamic_c_index_mean(data, "RISK", "EVENT", "TIME", "start", alpha=0.95, iterations=1000, n_jobs=50)
    avg = dynamic_c_index_avg("dynamic", data, "RISK", "EVENT", "TIME", "start")
    print(f"{avg} {ci}")

    out_file = f"experiments/results/model_{dataset}_mean_tdci_Apr_09_2023.txt"

    with open(out_file, "a") as f:
        f.write(f"rnn {feature}: {avg}, {ci} \n")


def tdci_matrix(feature: str, dataset: str):
    seed_everything(42)
    alpha = 0.95
    NUM_ITERATIONS = 1 # TODO - change to 1000


if __name__ == "__main__":
    import fire

    fire.Fire()