import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load, dump

DATA_DIR = ""


def add_covariate_to_timeline(df_static, df_dynamic, id_col, event_col, duration_col, outcome, delta = False):

    small_dfs = []

    for i, id in tqdm(enumerate(df_static[id_col]), desc=f'Merge static and dynamic ({outcome})', total=len(df_static)):
        df = df_dynamic[df_dynamic[id_col] == id].copy(deep=False)

        if len(df) <= 1:
            continue
        
        df_sta = df_static[df_static[id_col] == id]
        
        time = list(df_sta.TIME)[0]

        df = df[df[duration_col] <= time]
        if len(df) <= 0:
            continue

        if delta:
            df_temp = df.drop(columns=[id_col, duration_col])
            df_temp = df_temp.diff().fillna(0)

            # df_temp = df.diff()
            # df_temp = df_temp.iloc[:, 2:].div(df_temp[duration_col], axis=0).fillna(0)

            df_temp.rename(columns=lambda x: x+"_delta", inplace=True)
            df = pd.concat([df, df_temp], axis=1)

        df = df.merge(df_sta.drop(columns=["TIME"]), on="PX_ID", how="left")
        df = df.set_index(duration_col).sort_index().reset_index()

        df.rename(columns={duration_col:"stop"}, inplace=True)
        if time > df.loc[df.index[-1], "stop"]:
            df.loc[df.index[-1], "stop"] = time

        df[event_col] = [0] * len(df)
        df.loc[df["stop"] >= time, event_col] = int(df_sta.EVENT)

        start = list(df["stop"])
        del start[-1]
        start = [0] + start 
        df["start"] = start

        # remove duplicate rows for one patient
        df = df[~(df.stop == df.start)]

        small_dfs.append(df)

    new_df = pd.concat(small_dfs, ignore_index=True)

    new_df = new_df.loc[~((new_df["start"] == new_df["stop"]) & (new_df["start"] == 0))]

    return new_df


def process_iid(df, static_df):
    iid_df = df.drop(columns=["EVENT", "start"]).merge(
        static_df[["PX_ID", "EVENT", "TIME"]], on="PX_ID", how="left")

    iid_df["TIME"] = iid_df["TIME"] - iid_df["stop"]
    iid_df.rename(columns={"TIME":"T", "EVENT":"E"}, inplace=True)
    return iid_df.drop(columns=["PX_ID", "stop"])


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


