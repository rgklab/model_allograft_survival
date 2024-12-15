import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_dcox_feature_importance, dynamic_c_index_avg, load_OPTN_train

from mas.data.load_OPTN import load_OPTN_dataset
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_dynamic_cox import load_dynamic_cox, load_dynamic_cox_log, load_dynamic_cox_mice
from mas.data.load_dcox_delta import load_dcox_delta
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from joblib import load, dump

DATA_DIR = ""


def train(outcome: str, feature:str, k: int = 0, acc: int = 0, imputation: str = "ff"):

    out_file = f"experiments/results/{outcome}/dcox_{feature}.txt"

    penalizers = [0.1, 0.2, 0.5, 1, 10]  # with 0.01 before
    # penalizers = [0.2, 0.5, 1, 10, 100]
    l1_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    with open(out_file, "a") as f:
        f.write(f"{feature}_k={k}_acc={acc}_imputation={imputation}\n")

    _, graft_val_sta, _ = load_SRTR_static_df(outcome)

    if k == 0 and acc == 0:
        train, val, _ = load_dynamic_cox(outcome, imputation)
    elif k == 1:
        train, val, _ = load_dynamic_cox(outcome, "ff", delta=True)
    else:
        train, val, _ = load_dcox_delta(outcome, "ff", k=k, acc=True)

    if acc == 0:
        train = train.drop(columns=list((train.filter(regex="acc"))))
        val = val.drop(columns=list((val.filter(regex="acc"))))

    if feature != "full":

        train_info = train[["PX_ID", "EVENT", "start", "stop"]]
        val_info = val[["PX_ID", "EVENT", "start", "stop"]]
        if feature == "mas":
            variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        elif feature == "meld":
            variables = ["TFL_CREAT", "TFL_INR", "TFL_TOT_BILI"]
        elif feature == "meaf":
            variables = ["TFL_SGPT", "TFL_INR", "TFL_TOT_BILI"]
        elif feature == "albi":
            variables = ["TFL_TOT_BILI", "TFL_ALBUMIN"]

        reg=""
        for var in variables:
            reg += var+"|"
        reg = reg[:-1]
        train = pd.concat([train_info, train.filter(regex=reg)], axis=1)
        val = pd.concat([val_info, val.filter(regex=reg)], axis=1)

    print(train.shape)
    print(val.shape)

    print("training started")
    for penalizer in penalizers:
        for l1_ratio in l1_ratios:
            with open(out_file, "a") as f:
                f.write(f"MODEL: penalizer={penalizer}, l1_ratio={l1_ratio}\n")
            ct1 = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            try:
                ct1.fit(train, id_col="PX_ID", event_col="EVENT", start_col="start", stop_col="stop", show_progress=True)
            except:
                continue
            # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

            print("evaluation started")
            c_index = concordance_index(val["stop"], -ct1.predict_partial_hazard(val), val["EVENT"])

            # eval tcdi avg
            dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
            dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
            dcox_graft_df["RISK"] = ct1.predict_partial_hazard(val)

            avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
            with open(out_file, "a") as f:
                f.write(f"Validation C-Index: {c_index}; tcdi_avg: {avg} \n")
            print(c_index)
            print(f"tcdi avg: {avg}")


def train_mice(outcome: str = "graft", feature: str = "mas", imputation: str = "mice", dataset: int = 0):

    out_file = f"experiments/results/{outcome}/dcox_{feature}_{imputation}.txt"

    penalizers = [0.1, 0.2, 0.5, 1, 10]  # with 0.01 before
    # penalizers = [0.2, 0.5, 1, 10, 100]
    l1_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    with open(out_file, "a") as f:
        f.write(f"{feature}_imputation={imputation}_dataset={dataset}\n")

    _, graft_val_sta, _ = load_SRTR_static_df(outcome)

    train, val, _ = load_dynamic_cox_mice(outcome, imputation, dataset)

    if feature != "full":

        train_info = train[["PX_ID", "EVENT", "start", "stop"]]
        val_info = val[["PX_ID", "EVENT", "start", "stop"]]
        if feature == "mas":
            variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        elif feature == "meld":
            variables = ["TFL_CREAT", "TFL_INR", "TFL_TOT_BILI"]
        elif feature == "meaf":
            variables = ["TFL_SGPT", "TFL_INR", "TFL_TOT_BILI"]
        elif feature == "albi":
            variables = ["TFL_TOT_BILI", "TFL_ALBUMIN"]

        reg=""
        for var in variables:
            reg += var+"|"
        reg = reg[:-1]
        train = pd.concat([train_info, train.filter(regex=reg)], axis=1)
        val = pd.concat([val_info, val.filter(regex=reg)], axis=1)

    print(train.shape)
    print(val.shape)

    print("training started")
    for penalizer in penalizers:
        for l1_ratio in l1_ratios:
            with open(out_file, "a") as f:
                f.write(f"MODEL: penalizer={penalizer}, l1_ratio={l1_ratio}\n")
            ct1 = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            try:
                ct1.fit(train, id_col="PX_ID", event_col="EVENT", start_col="start", stop_col="stop", show_progress=True)
            except:
                continue
            # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

            print("evaluation started")
            c_index = concordance_index(val["stop"], -ct1.predict_partial_hazard(val), val["EVENT"])

            # eval tcdi avg
            dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
            dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
            dcox_graft_df["RISK"] = ct1.predict_partial_hazard(val)

            avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
            with open(out_file, "a") as f:
                f.write(f"Validation C-Index: {c_index}; tcdi_avg: {avg} \n")
            print(c_index)
            print(f"tcdi avg: {avg}")


def final_model(outcome: str, feature:str, k: int = 0, acc: int = 0):

    # mortality full
    # MODEL: penalizer=0.1, l1_ratio=0
    # Validation C-Index: 0.8768945793277293; tcdi_avg: 0.7690411934640983 

    # mortality mas
    # MODEL: penalizer=0.5, l1_ratio=0
    # Validation C-Index: 0.8207802136520891; tcdi_avg: 0.7310952802425237 

    # MELD ABC
    # MODEL: penalizer=0.1, l1_ratio=0.3
    # Validation C-Index: 0.8762301461037687; tcdi_avg: 0.7882459969960514 

    # MEAF ABC
    # MODEL: penalizer=0.1, l1_ratio=0.5
    # Validation C-Index: 0.8801553089638465; tcdi_avg: 0.7931985934063109 

    # ALBI ABC
    # MODEL: penalizer=0.5, l1_ratio=0.1
    # Validation C-Index: 0.9145486239759303; tcdi_avg: 0.833132253021285 

    out_file = f"experiments/models/dynamic_cox/{outcome}/{feature}_k={k}_acc={acc}.pkl"


    _, graft_val_sta, _ = load_SRTR_static_df(outcome)

    if k == 0 and acc == 0:
        train, val, _ = load_dynamic_cox(outcome, "ff")
    elif k == 1:
        train, val, _ = load_dynamic_cox(outcome, "ff", delta=True)
    else:
        train, val, _ = load_dcox_delta(outcome, "ff", k=k, acc=True)

    if acc == 0:
        train = train.drop(columns=list((train.filter(regex="acc"))))
        val = val.drop(columns=list((val.filter(regex="acc"))))

    train_info = train[["PX_ID", "EVENT", "start", "stop"]]
    val_info = val[["PX_ID", "EVENT", "start", "stop"]]
    if feature == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
    elif feature == "meld":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_TOT_BILI"]
    elif feature == "meaf":
        variables = ["TFL_SGPT", "TFL_INR", "TFL_TOT_BILI"]
    elif feature == "albi":
        variables = ["TFL_TOT_BILI", "TFL_ALBUMIN"]

    reg=""
    for var in variables:
        reg += var+"|"
    reg = reg[:-1]
    train = pd.concat([train_info, train.filter(regex=reg)], axis=1)
    val = pd.concat([val_info, val.filter(regex=reg)], axis=1)

    print(train.shape)
    print(val.shape)

    print("training started")
 
    ct1 = CoxTimeVaryingFitter(penalizer=0.5, l1_ratio=0.1)
    ct1.fit(train, id_col="PX_ID", event_col="EVENT", start_col="start", stop_col="stop", show_progress=True)
  
    dump(ct1, out_file)

    print("evaluation started")
    c_index = concordance_index(val["stop"], -ct1.predict_partial_hazard(val), val["EVENT"])

    # eval tcdi avg
    dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    dcox_graft_df["RISK"] = ct1.predict_partial_hazard(val)

    avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
    print(c_index)
    print(f"tcdi avg: {avg}")



def evaluate(k: int, acc: int, feature: str, model_path: 
                str = "experiments/models/dynamic_models_pkl/risk_score/mas_abc.pkl"):

    _, graft_val_sta, _ = load_SRTR_static_df("graft")

    if k == 0 and acc == 0:
        train, val, _ = load_dynamic_cox("graft", "ff")
    else:
        train, val, _ = load_dcox_delta("graft", "ff", k=k, acc=True)
        model_path = f"experiments/models/dynamic_cox/graft_{feature}_k={k}_acc={acc}.pkl"

    model = load(model_path)

    if acc == 0:
        train = train.drop(columns=list((train.filter(regex="acc"))))
        val = val.drop(columns=list((val.filter(regex="acc"))))

    if feature == "mas":
        train_info = train[["PX_ID", "EVENT", "start", "stop"]]
        val_info = val[["PX_ID", "EVENT", "start", "stop"]]
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]; reg=""
        for var in variables:
            reg += var+"|"
        reg = reg[:-1]
        train = pd.concat([train_info, train.filter(regex=reg)], axis=1)
        val = pd.concat([val_info, val.filter(regex=reg)], axis=1)

    
    print("evaluation started")
    c_index = concordance_index(val["stop"], -model.predict_partial_hazard(val), val["EVENT"])

    # eval tcdi avg
    dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    dcox_graft_df["RISK"] = model.predict_partial_hazard(val)

    avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
    print(c_index)
    print(f"tcdi avg: {avg}")


def train_OPTN(OPTN: int, outcome:str, feature:str):
    out_file = f"experiments/results/{outcome}/OPTN/dcox_region-{OPTN}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{OPTN}\n")

    # wrong data split
    # _, graft_val_sta = load_OPTN_dataset(OPTN, outcome, "static")
    # train, val = load_OPTN_dataset(OPTN, outcome, "dynamic")

    train, val, test = load_OPTN_train(OPTN, "dynamic", outcome)
    train_sta, val_sta, test_sta = load_OPTN_train(OPTN, "static", outcome)
    # use both val and test as one set due to small sample size
    val = pd.concat([val, test])
    graft_val_sta = pd.concat([val_sta, test_sta])

    penalizers = [0.01, 0.1, 0.2, 0.5, 1, 10]
    # penalizers = [0.2, 0.5, 1, 10, 100]
    l1_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    if feature == "full":
        pass
    elif feature == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        train = train[["PX_ID", "EVENT", "start", "stop"] + variables]
        val = val[["PX_ID", "EVENT", "start", "stop"] + variables]

    print(train.shape)
    print(val.shape)

    print("training started")
    for penalizer in penalizers:
        for l1_ratio in l1_ratios:
            with open(out_file, "a") as f:
                f.write(f"MODEL: penalizer={penalizer}, l1_ratio={l1_ratio}\n")
            ct1 = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            try:
                ct1.fit(train, id_col="PX_ID", event_col="EVENT", start_col="start", stop_col="stop", show_progress=True)
            except:
                continue
            # dump(ct1, f"experiments/models/dynamic_cox/{outcome}_{feature}_k={k}_acc={acc}.pkl")

            print("evaluation started")
            c_index = concordance_index(val["stop"], -ct1.predict_partial_hazard(val), val["EVENT"])

            # eval tcdi avg
            dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
            dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
            dcox_graft_df["RISK"] = ct1.predict_partial_hazard(val)

            avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
            with open(out_file, "a") as f:
                f.write(f"Validation C-Index: {c_index}; tcdi_avg: {avg} \n")
            print(c_index)
            print(f"tcdi avg: {avg}")



if __name__ == "__main__":
    import fire

    fire.Fire()