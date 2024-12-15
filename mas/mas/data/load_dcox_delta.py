import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
from mas.data.load_data import load_SRTR_static, load_SRTR_static_df
from mas.data.load_data_dynamic import load_SRTR_dynamic
from tqdm import tqdm

DATA_DIR = ""

def load_dcox_delta(
    outcome : str,
    imputation: str,
    k: int = 1,
    acc: bool = False,
    load_from_pkl : bool = True,
    normalized: bool = True
    ):
    """
    Loads and interpolates dynamic time series data from TXF_LI and FOL_IMMUNO table.

    """
    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/train_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")
        val = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/val_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")
        test = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/test_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")

        return (train, val, test)
        
    else:
        train_dynamic, val_dynamic, test_dynamic = load_SRTR_dynamic(imputation = imputation)

        validator = DatasetValidator()
        if normalized:
            train_dynamic, val_dynamic, test_dynamic = validator.validate_dataframe(
                train_dynamic, val_dynamic, test_dynamic)

            tr_dy = train_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
            tr_ids, tr_time = train_dynamic.PX_ID, train_dynamic.TIME_SINCE_BASELINE

            train_mean = np.average(tr_dy, axis=0); train_std = np.std(tr_dy, axis=0)

            train_dynamic = (tr_dy - train_mean) / (train_std)
            train_dynamic["PX_ID"], train_dynamic["TIME_SINCE_BASELINE"] = tr_ids, tr_time

            va_dy = val_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
            va_ids, va_time = val_dynamic.PX_ID, val_dynamic.TIME_SINCE_BASELINE
            val_dynamic = (va_dy - train_mean) / (train_std)
            val_dynamic["PX_ID"], val_dynamic["TIME_SINCE_BASELINE"] = va_ids, va_time

            te_dy = test_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
            te_ids, te_time = test_dynamic.PX_ID, test_dynamic.TIME_SINCE_BASELINE
            test_dynamic = (te_dy - train_mean) / (train_std)
            test_dynamic["PX_ID"], test_dynamic["TIME_SINCE_BASELINE"] = te_ids, te_time

        train_df_static, val_df_static, test_df_static = load_SRTR_static_df(outcome)

        train = _add_covariate_to_timeline(train_df_static, 
            train_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, k=k, acc=acc)
        val = _add_covariate_to_timeline(val_df_static, 
            val_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, k=k, acc=acc)
        test = _add_covariate_to_timeline(test_df_static, 
            test_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, k=k, acc=acc)

        train, val, test = validator.validate_dataframe(train, val, test)

        train.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/train_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")
        val.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/val_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")
        test.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dcox_delta/test_dcox_{outcome}_{acc}_{k}_norm:{normalized}.pkl")

        return (train, val, test)

def _add_covariate_to_timeline(df_static, df_dynamic, id_col, event_col, duration_col, outcome, k = 1, acc = False):
    if k == 1 and acc is True:
        raise Exception("Going back one step does not provide acceleration!")

    small_dfs = []

    for i, id in tqdm(enumerate(df_static[id_col]), desc=f'Merge static and dynamic ({outcome})', total=len(df_static)):
        df = df_dynamic[df_dynamic[id_col] == id].copy(deep=False)
        df.reset_index(drop=True, inplace=True)

        if len(df) <= 1:
            continue
        
        df_sta = df_static[df_static[id_col] == id]
        
        time = list(df_sta.TIME)[0]

        df = df[df[duration_col] <= time]
        if len(df) <= 0:
            continue

        #### deal with delta
        df_temp = df.drop(columns=[id_col, duration_col])
        df_temp.reset_index(drop=True, inplace=True)
        for i in range(1, k+1):
            if i == 1:
                new_df = df_temp.diff().fillna(0)
                new_df.rename(columns=lambda x: x+"_delta"+"_1", inplace=True)
                new_df = pd.concat([df, new_df], axis=1)
                new_df.reset_index(drop=True, inplace=True)
            else:
                # start moving diffs 
                diff = df_temp.diff().fillna(0)
                new_diff = pd.concat([pd.DataFrame(0, index=np.arange(i-1), columns=diff.columns), diff.iloc[:-(i-1), :]])
                new_diff = new_diff.iloc[:len(df_temp), :]
                new_diff.rename(columns=lambda x: x+"_delta"+f"_{i}", inplace=True)
                new_diff.reset_index(drop=True, inplace=True)
                new_df = pd.concat([new_df, new_diff], axis=1)

        if acc:
            for i in range(1, k):
                for column in df_temp.columns:
                    new_df[column+"_acc"+f"_{i}"] = new_df[column+"_delta"+f"_{i}"] - new_df[column+"_delta"+f"_{i+1}"]
        ####

        new_df = new_df.merge(df_sta.drop(columns=["TIME"]), on="PX_ID", how="left")
        new_df = new_df.set_index(duration_col).sort_index().reset_index()

        new_df.rename(columns={duration_col:"stop"}, inplace=True)
        if time > new_df.loc[new_df.index[-1], "stop"]:
            new_df.loc[new_df.index[-1], "stop"] = time

        new_df[event_col] = [0] * len(new_df)
        new_df.loc[new_df["stop"] >= time, event_col] = int(df_sta.EVENT)

        start = list(new_df["stop"])
        del start[-1]
        start = [0] + start 
        new_df["start"] = start

        small_dfs.append(new_df)

    new_df = pd.concat(small_dfs, ignore_index=True)

    new_df = new_df.loc[~((new_df["start"] == new_df["stop"]) & (new_df["start"] == 0))]

    return new_df


class DatasetValidator():

    def validate_dataframe(self, train_set, val_set, test_set):
        """
        1. Check the splits for NaN; drop rows/fill accordingly.
        """
        # Check the splits for things that would yield a NaN - specifically,
        # check if there are any columns with only one unique value; these go
        # to infinity when normalizing.

        cols_to_drop = []
        for i in range(train_set.shape[1]):  # Loop through columns
            if len(train_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(val_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(test_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        train_set = train_set.iloc[:, [col for col in range(train_set.shape[1]) if col not in cols_to_drop]]
        val_set = val_set.iloc[:, [col for col in range(val_set.shape[1]) if col not in cols_to_drop]]
        test_set = test_set.iloc[:, [col for col in range(test_set.shape[1]) if col not in cols_to_drop]]

        return train_set, val_set, test_set


if __name__ == "__main__":
    # load_dynamic_cox("graft", False, normalized=False)
    # load_dynamic_cox("mortality", False, normalized=False)

    # load_dynamic_cox("graft", "mice", False)
    # load_dynamic_cox("graft", "mice", False, delta=True)

    # load_dynamic_cox("graft", "mice_strata", False)
    # load_dynamic_cox("graft", "mice_strata", False, delta=True)

    # load_dynamic_cox("graft", "mice_gbdt", False)
    # load_dynamic_cox("graft", "mice_gbdt", False, delta=True)

    # load_dynamic_cox("graft", "ff", False)
    # load_dcox_delta("graft", "ff", k=2, acc=True, load_from_pkl=False)
    load_dcox_delta("graft", "ff", k=3, acc=True, load_from_pkl=False)