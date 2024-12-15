import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
from mas.data.load_data import load_SRTR_static, load_SRTR_static_df
from mas.data.utils import add_covariate_to_timeline, DatasetValidator
from mas.data.load_data_dynamic import load_SRTR_dynamic
from tqdm import tqdm

DATA_DIR = ""

def load_dynamic_cox(
    outcome : str,
    imputation: str,
    load_from_pkl : bool = True,
    feature: str = "all",
    delta : bool = False,
    normalized: bool = True,
    excluding_first_year: bool = False
    ):
    """
    Loads and interpolates dynamic time series data from TXF_LI and FOL_IMMUNO table.

    """
    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")
        val = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")
        test = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")

        if excluding_first_year:
            train_sta, val_sta, test_sta = load_SRTR_static_df("graft")
            ids_train = train_sta[(train_sta.EVENT == 1) & (train_sta.TIME < 1)].PX_ID
            ids_val = val_sta[(val_sta.EVENT == 1) & (val_sta.TIME < 1)].PX_ID
            ids_test = test_sta[(test_sta.EVENT == 1) & (test_sta.TIME < 1)].PX_ID

            train = train[~train.PX_ID.isin(ids_train)]
            val = val[~val.PX_ID.isin(ids_val)]
            test = test[~test.PX_ID.isin(ids_test)]
                
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
            train_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, delta=delta)
        val = _add_covariate_to_timeline(val_df_static, 
            val_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, delta=delta)
        test = _add_covariate_to_timeline(test_df_static, 
            test_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome, delta=delta)

        train, val, test = validator.validate_dataframe(train, val, test)

        train.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")
        val.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")
        test.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_{feature}_delta:{delta}_norm:{normalized}.pkl")

        return (train, val, test)

def _add_covariate_to_timeline(df_static, df_dynamic, id_col, event_col, duration_col, outcome, delta = False):

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


def load_dynamic_cox_log(
    outcome : str,
    imputation: str,
    load_from_pkl : bool = True,
    feature: str = "all",
    ):
    """
    Loads and interpolates dynamic time series data from TXF_LI and FOL_IMMUNO table.

    """
    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_{feature}_log.pkl")
        val = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_{feature}_log.pkl")
        test = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_{feature}_log.pkl")
                
        return (train, val, test)
        
    else:
        train_dynamic, val_dynamic, test_dynamic = load_SRTR_dynamic(imputation = imputation)


        tr_dy = train_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
        tr_ids, tr_time = train_dynamic.PX_ID, train_dynamic.TIME_SINCE_BASELINE
        train_dynamic = np.log(tr_dy+1)
        train_dynamic["PX_ID"], train_dynamic["TIME_SINCE_BASELINE"] = tr_ids, tr_time

        va_dy = val_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
        va_ids, va_time = val_dynamic.PX_ID, val_dynamic.TIME_SINCE_BASELINE
        val_dynamic = np.log(va_dy+1)
        val_dynamic["PX_ID"], val_dynamic["TIME_SINCE_BASELINE"] = va_ids, va_time

        te_dy = test_dynamic.drop(columns=["PX_ID", "TIME_SINCE_BASELINE"])
        te_ids, te_time = test_dynamic.PX_ID, test_dynamic.TIME_SINCE_BASELINE
        test_dynamic = np.log(te_dy+1)
        test_dynamic["PX_ID"], test_dynamic["TIME_SINCE_BASELINE"] = te_ids, te_time

        train_df_static, val_df_static, test_df_static = load_SRTR_static_df(outcome)

        train = _add_covariate_to_timeline(train_df_static, 
            train_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)
        val = _add_covariate_to_timeline(val_df_static, 
            val_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)
        test = _add_covariate_to_timeline(test_df_static, 
            test_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)


        train.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_{feature}_log.pkl")
        val.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_{feature}_log.pkl")
        test.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_{feature}_log.pkl")

        return (train, val, test)
    


def load_dynamic_cox_mice(
    outcome : str = "graft",
    imputation: str = "mice",
    dataset: int = 0,
    load_from_pkl : bool = True,
    normalized: bool = True
    ):
    """
    For Multiple Imputation dataset, dataset goes from 0 to 4.
    """
    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")
        val = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")
        test = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")
                
        return (train, val, test)
        
    else:
        train_dynamic, val_dynamic, test_dynamic = load_SRTR_dynamic(imputation = imputation, dataset=dataset)

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
            train_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)
        val = _add_covariate_to_timeline(val_df_static, 
            val_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)
        test = _add_covariate_to_timeline(test_df_static, 
            test_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome=outcome)

        train, val, test = validator.validate_dataframe(train, val, test)

        train.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")
        val.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")
        test.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_{outcome}_norm:{normalized}_{dataset}.pkl")



if __name__ == "__main__":
    import fire

    fire.Fire()