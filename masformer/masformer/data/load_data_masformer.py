import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load, dump

from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.data.load_OPTN import load_OPTN_dataset


def load_data_masformer(
    outcome : str,
    variables: str = "full",
    load_from_pkl : bool = True,
    feature: str = "all"
    ):
    if load_from_pkl:
        train = load(f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/train_{outcome}_{feature}.pkl")
        val = load(f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/val_{outcome}_{feature}.pkl")
        test = load(f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/test_{outcome}_{feature}.pkl")

        return (train, val, test)
        
    else:
        train, val, test = load_dynamic_cox(outcome=outcome, imputation="ff")

        # val = val[val.PX_ID != 470812] # 470812

        max_duration = max(train.groupby("PX_ID").size().max() - 1, val.groupby("PX_ID").size().max() - 1)
        max_duration = max(test.groupby("PX_ID").size().max() - 1, max_duration)

        print(max_duration) # print 19 actual 20 (plus one later)

        train = _get_features_and_labels(train, variables, max_duration)
        val = _get_features_and_labels(val, variables, max_duration)
        test = _get_features_and_labels(test, variables, max_duration)

        dump(train, f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/train_{outcome}_{feature}.pkl")
        dump(val, f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/val_{outcome}_{feature}.pkl")
        dump(test, f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/test_{outcome}_{feature}.pkl")

        return (train, val, test)


def load_optn_eval_masformer(
    outcome : str,
    variables: str,
    OPTN: int,
    load_from_pkl : bool = True
    ):
    if load_from_pkl:
        val = load(f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/val_{outcome}_{OPTN}.pkl")
        test = load(f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/test_{outcome}_{OPTN}.pkl")

        return (val, test)
        
    else:

        val, test = _load_optn_eval(OPTN, "dynamic", outcome)

        # max_duration = max(train.groupby("PX_ID").size().max() - 1, val.groupby("PX_ID").size().max() - 1)
        # max_duration = max(test.groupby("PX_ID").size().max() - 1, max_duration)

        # print(max_duration)
        max_duration = 19 # actual is 20, here is minus 1

        val = _get_features_and_labels(val, variables, max_duration)
        test = _get_features_and_labels(test, variables, max_duration)

        dump(val, f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/val_{outcome}_{OPTN}.pkl")
        dump(test, f"masformer/data/dynamic_postprocessed/ff/masformer/{variables}/test_{outcome}_{OPTN}.pkl")

        return (val, test)


def _get_features_and_labels(df, variables, max_duration):
    features, labels = [], []
    # max_duration = df.groupby("PX_ID").size().max() - 1
    ids = df.PX_ID.unique()

    if variables == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        df = df[["PX_ID", "EVENT", "start", "stop"] + variables]

    for id in tqdm(ids):
        id_df = df[df.PX_ID == id].copy()
        id_df["idx"] = np.arange(0, len(id_df))
        if id_df.EVENT.sum() > 0:
            duration = int(id_df[id_df.EVENT == 1].drop_duplicates(subset=['EVENT'], keep='first').idx)
            time = float(id_df[id_df.EVENT == 1].drop_duplicates(subset=['EVENT'], keep='first').stop)
            is_observed = 1
        else:
            duration = int(id_df[id_df.EVENT == 0].drop_duplicates(subset=['EVENT'], keep='last').idx)
            time = float(id_df[id_df.EVENT == 0].drop_duplicates(subset=['EVENT'], keep='last').stop)
            is_observed = 0
        labels.append([time, duration, is_observed])
        num_pad = max_duration - duration
        id_df = id_df.iloc[:duration+1]
        id_df.drop(columns=["stop", "start", "EVENT", "PX_ID", "idx"], inplace=True)
        id_arr = np.asarray(id_df)
        padding = np.full((num_pad, id_arr.shape[1]), np.nan)
        features.append(np.vstack((id_arr, padding)))
    return (np.asarray(features), np.asarray(labels))


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
    # load_data_masformer("graft", "full", False)
    # load_data_masformer("graft", "mas", False)
    # load_data_masformer("cancer", "full", False)
    # load_data_masformer("cancer", "mas", False)

    for i in range(1, 12):
        load_optn_eval_masformer("graft", "full", i, False)
        load_optn_eval_masformer("graft", "mas", i, False)