import os

from torch import normal
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import theano
import lasagne
print(f"Device: {theano.config.device}")

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_dcox_feature_importance, dynamic_c_index_avg

from mas.models.deepsurv import DeepSurv
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_OPTN import load_OPTN_dataset
from mas.data.load_PD_group import load_PD_dataset
from mas.data.load_data_deepsurv import load_deepsurv
from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_dcox_delta import load_dcox_delta
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from joblib import load, dump


def _dict_product(dicts):
    """
    Credit: https://stackoverflow.com/a/40623158/7978975
    """
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def train(outcome:str, feature:str, normalized = True):
    lasagne.random.set_rng(np.random.RandomState(51))
    normalized = bool(normalized)
    out_file = f"experiments/results/{outcome}/deepsurv_{feature}_{normalized}.txt"

    hparams = {}
    hparams["hidden_layers_sizes"] = [
                                                [16, 16],
                                                [64, 64],
                                                [128, 128],
                                                [512, 512],
                                                [16, 16, 16],
                                                [128, 128, 128],
                                                [512, 512, 512],
                                                [16, 16, 16, 16],
                                                [64, 64, 64, 64],
                                                [512, 512, 512, 512],
                                            ]
    hparams["learning_rate"] = [1e-4, 5e-4, 1e-3] 
    hparams["dropout"] = [0, 0.1, 0.3] 
    hparams["learning_rate"] = [1e-4, 5e-4, 1e-3] 
    hparams["L2_reg"] = [0, 0.5, 1, 10] 

    deepsurv_train, deepsurv_val, _ = load_deepsurv(outcome, feature, normalized)
    _, val_sta, _ = load_SRTR_static_df(outcome)
    train, val, _ = load_dynamic_cox(outcome, "ff")

    hparams["n_in"] = [deepsurv_train['x'].shape[1]]
    print(hparams["n_in"])

    hparams = _dict_product(hparams)

    print("training started")
    for hparam in hparams:
        lasagne.random.set_rng(np.random.RandomState(51))
        with open(out_file, "a") as f:
            f.write(f"MODEL: {str(hparam)}\n")
        model = DeepSurv(**hparam)
        try:
            model.train(deepsurv_train, valid_data=deepsurv_val, n_epochs=1000)
        except:
            continue
        # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

        print("evaluation started")
        try:
            with open(out_file, "a") as f:
                cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
                deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
                deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
                deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
                avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
                f.write(f"Validation C-Index: {cindex}; tcdi_avg: {avg} \n")
                print(cindex)
                print(f"tcdi avg: {avg}")
        except ValueError:
            with open(out_file, "a") as f:
                f.write("NaN\n")
                print("NaN")


def final_model(outcome:str, feature:str, normalized = True):

    # mortality full
    # MODEL: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
    # Validation C-Index: 0.7274062314547559; tcdi_avg: 0.7629058514030125 

    # mortality mas
    # MODEL: {'hidden_layers_sizes': [512, 512, 512], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 10, 'n_in': 6}
    # Validation C-Index: 0.695252919510163; tcdi_avg: 0.7366367291570494 


    normalized = bool(normalized)
    out_file = f"experiments/models/deepmas/{outcome}/{feature}_{normalized}.pkl"

    deepsurv_train, deepsurv_val, _ = load_deepsurv(outcome, feature, normalized)
    _, val_sta, _ = load_SRTR_static_df(outcome)
    train, val, _ = load_dynamic_cox(outcome, "ff")

    print(deepsurv_train['x'].shape[1])

    hparam = {'hidden_layers_sizes': [512, 512, 512], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 10, 'n_in': 6}
    print("training started")
    lasagne.random.set_rng(np.random.RandomState(51))
    model = DeepSurv(**hparam)
    model.train(deepsurv_train, valid_data=deepsurv_val, n_epochs=1000)
    dump(model, out_file)

    print("evaluation started")
    cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
    deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
    deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
    avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
    print(cindex)
    print(f"tcdi avg: {avg}")


def train_PD(PD: str, outcome:str, feature:str):
    lasagne.random.set_rng(np.random.RandomState(51))
    out_file = f"experiments/results/{outcome}/PD/deepsurv_{PD}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{PD}\n")

    hparams = {}
    hparams["hidden_layers_sizes"] = [
                                                [16, 16],
                                                [64, 64],
                                                [128, 128],
                                                [512, 512],
                                                [16, 16, 16],
                                                [128, 128, 128],
                                                [512, 512, 512],
                                                [16, 16, 16, 16],
                                                [64, 64, 64, 64],
                                                [512, 512, 512, 512],
                                            ]
    hparams["learning_rate"] = [1e-4, 5e-4, 1e-3] 
    hparams["dropout"] = [0, 0.1, 0.3] 
    hparams["learning_rate"] = [1e-4, 5e-4, 1e-3] 
    hparams["L2_reg"] = [0, 0.5, 1, 10] 

    deepsurv_train, deepsurv_val, _ = load_deepsurv(outcome, feature, PD=PD)
    _, val_sta, _ = load_PD_dataset(PD, outcome, "static")
    train, val, _ = load_PD_dataset(PD, outcome, "dynamic")

    hparams["n_in"] = [deepsurv_train['x'].shape[1]]
    print(hparams["n_in"])

    hparams = _dict_product(hparams)

    print("training started")
    for hparam in hparams:
        with open(out_file, "a") as f:
            f.write(f"MODEL: {str(hparam)}\n")
        model = DeepSurv(**hparam)
        try:
            model.train(deepsurv_train, valid_data=deepsurv_val, n_epochs=1000)
        except:
            continue
        # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

        print("evaluation started")
        try:
            with open(out_file, "a") as f:
                cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
                deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
                deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
                deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
                avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
                f.write(f"Validation C-Index: {cindex}; tcdi_avg: {avg} \n")
                print(cindex)
                print(f"tcdi avg: {avg}")
        except ValueError:
            with open(out_file, "a") as f:
                f.write("NaN\n")
                print("NaN")


def train_OPTN(OPTN: int, outcome:str, feature:str):
    lasagne.random.set_rng(np.random.RandomState(51))
    out_file = f"experiments/results/{outcome}/OPTN/deepsurv_region-{OPTN}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{OPTN}\n")

    hparams = {}
    hparams["hidden_layers_sizes"] = [
                                                [16, 16],
                                                [64, 64],
                                                [128, 128],
                                                [512, 512],
                                                [16, 16, 16],
                                                [128, 128, 128],
                                                [512, 512, 512],
                                                [16, 16, 16, 16],
                                                [64, 64, 64, 64],
                                                [512, 512, 512, 512],
                                            ]
    hparams["learning_rate"] = [1e-4, 5e-4, 1e-3] 
    hparams["dropout"] = [0, 0.1, 0.3] 
    hparams["L2_reg"] = [0, 0.5, 1, 10] 

    deepsurv_train, deepsurv_val = load_deepsurv(outcome, feature, OPTN=OPTN)
    _, val_sta = load_OPTN_dataset(OPTN, outcome, "static")
    train, val = load_OPTN_dataset(OPTN, outcome, "dynamic")

    hparams["n_in"] = [deepsurv_train['x'].shape[1]]
    print(hparams["n_in"])

    hparams = _dict_product(hparams)

    print("training started")
    for hparam in hparams:
        lasagne.random.set_rng(np.random.RandomState(51))
        with open(out_file, "a") as f:
            f.write(f"MODEL: {str(hparam)}\n")
        model = DeepSurv(**hparam)
        try:
            model.train(deepsurv_train, valid_data=deepsurv_val, n_epochs=1000)
        except:
            continue
        # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

        print("evaluation started")
        try:
            with open(out_file, "a") as f:
                cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
                deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
                deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
                deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
                avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
                f.write(f"Validation C-Index: {cindex}; tcdi_avg: {avg} \n")
                print(cindex)
                print(f"tcdi avg: {avg}")
        except ValueError:
            with open(out_file, "a") as f:
                f.write("NaN\n")
                print("NaN")


if __name__ == "__main__":
    import fire

    fire.Fire()