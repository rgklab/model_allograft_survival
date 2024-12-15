import pandas as pd
import numpy as np
from tqdm import tqdm
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from mas.data.load_data import load_SRTR_static_df
from mas.data.load_data_dynamic import load_SRTR_dynamic
from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.utils import *
from mas.data.df_utils import *
from joblib import load, dump

DATA_DIR = ""
SPLIT_DIR = ""


def create_death_prior_gf():
    """
    Find patients who died due to causes other than graft failure 
    and their mortality time
    """
    tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    train,val,test = load_dynamic_cox("graft", "ff")
    dataset = pd.concat([train,val,test])
    ids = dataset.PX_ID.unique()

    tx_li = tx_li[(tx_li.PX_ID.isin(ids)) & (pd.notnull(tx_li["TFL_DEATH_DT"]))]
    tx_li = tx_li[["PX_ID", "TFL_DEATH_DT", "REC_TX_DT", "TFL_COD"]]

    # remove patients died due to graft failure
    tx_li = tx_li.loc[(tx_li["TFL_COD"] != 4600) &  
                (tx_li["TFL_COD"] != 4601) &
                (tx_li["TFL_COD"] != 4602) &
                (tx_li["TFL_COD"] != 4603) &
                (tx_li["TFL_COD"] != 4604) &
                (tx_li["TFL_COD"] != 4605) &
                (tx_li["TFL_COD"] != 4606) &
                (tx_li["TFL_COD"] != 4610) &
                (tx_li["TFL_COD"] != 4615) &
                (tx_li["TFL_COD"] != 4955) &
                (tx_li["TFL_COD"] != 4956) &
                (tx_li["TFL_COD"] != 4957) &
                (tx_li["TFL_COD"] != 4958)
                ]
    tx_li["TIME"] = ((tx_li["TFL_DEATH_DT"] - tx_li["REC_TX_DT"]).dt.days)/365
    dump(tx_li[["PX_ID", "TIME"]], "mas/data/reference_tables/death_prior_gf.csv")
    print("Dump the final results.")


def load_SRTR_static_gfcr():
    """
    LOAD the static dataset for graft failure with competing risks (death 
    prior to GF)
    """
    train_sta, val_sta, test_sta = load_SRTR_static_df("graft")
    dataset_sta =pd.concat([train_sta,val_sta,test_sta])
    ids = get_IDs()
    cr = load("mas/data/reference_tables/death_prior_gf.csv")

    join = dataset_sta.merge(cr, how="left", on="PX_ID")
    join = join[join.PX_ID.isin(ids)]
    event = join["EVENT"].copy()
    join["EVENT"] = join["EVENT"].where(~((pd.notnull(join["TIME_y"])) & (join["EVENT"]==0)), 2)
    join["TIME"] = join["TIME_x"].where(~((pd.notnull(join["TIME_y"])) & (event==0)), join["TIME_y"])
    join = join.drop(columns=["TIME_x", "TIME_y"])

    px_ids = {}
    for split in ["train", "val", "test"]:
        px_ids[split] = np.loadtxt(f"mas/data/data_splits/{split}_split.txt", delimiter='\n')

    train = join[join.PX_ID.isin(px_ids["train"])]
    val = join[join.PX_ID.isin(px_ids["val"])]
    test = join[join.PX_ID.isin(px_ids["test"])]

    return train, val, test


def load_dcox_gfcr(
    imputation: str = "ff",
    load_from_pkl : bool = True,
    delta : bool = False,
    normalized: bool = True
    ):
    """
    Loads and interpolates dynamic time series data from TXF_LI and FOL_IMMUNO table.
    (with competing risks)
    """
    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")
        val = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")
        test = pd.read_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")
                
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

        train_df_static, val_df_static, test_df_static = load_SRTR_static_gfcr()

        train = add_covariate_to_timeline(train_df_static, 
            train_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome="graft", delta=delta)
        val = add_covariate_to_timeline(val_df_static, 
            val_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome="graft", delta=delta)
        test = add_covariate_to_timeline(test_df_static, 
            test_dynamic, duration_col="TIME_SINCE_BASELINE", id_col="PX_ID", event_col="EVENT", outcome="graft", delta=delta)

        train, val, test = validator.validate_dataframe(train, val, test)

        train.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/train_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")
        val.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/val_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")
        test.to_pickle(f"mas/data/dynamic_postprocessed/{imputation}/dynamic_cox/test_dcox_gfcr_delta:{delta}_norm:{normalized}.pkl")

        return (train, val, test)


def load_OPTN_crgf(OPTN: int, mode: str):
    train, val, test = load_dcox_gfcr() if mode == "dynamic" else load_SRTR_static_gfcr()
    dataset =pd.concat([train, val, test])

    px_ids = {}
    for split in ["train", "test"]:
        px_ids[split] = np.loadtxt(f"{SPLIT_DIR}/OPTN/region_{OPTN}/{split}_split.txt", delimiter='\n')

    train = dataset[dataset.PX_ID.isin(px_ids["train"])]
    test = dataset[dataset.PX_ID.isin(px_ids["test"])]

    return train, test


def load_deephit(variables :str, normalized = True, OPTN = None):
    if OPTN:
        train, val, test = load_dcox_gfcr(normalized=normalized)
        train, test = load_OPTN_crgf(OPTN, "dynamic")
    else:
        train, val, test = load_dcox_gfcr(normalized=normalized)

    if variables == "full":
        pass
    elif variables == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        train = train[["PX_ID", "EVENT", "start", "stop"] + variables]
        val = val[["PX_ID", "EVENT", "start", "stop"] + variables]
        test = test[["PX_ID", "EVENT", "start", "stop"] + variables]
    else:
        train = train[["PX_ID", "EVENT", "start", "stop"] + variables]
        val = val[["PX_ID", "EVENT", "start", "stop"] + variables]
        test = test[["PX_ID", "EVENT", "start", "stop"] + variables]

    if OPTN:
        train_sta, val_sta, test_sta = load_SRTR_static_gfcr()
        train_sta, test_sta = load_OPTN_crgf(OPTN, "static")
    else:
        train_sta, val_sta, test_sta = load_SRTR_static_gfcr()

    deephit_train = process_iid(train, train_sta)
    deephit_val = process_iid(val, val_sta)
    deephit_test = process_iid(test, test_sta)

    get_x = lambda df: (df
                    .drop(columns=['T', 'E'])
                    .values.astype('float32'))
    get_target = lambda df: (df['T'].values, df['E'].values)
    # manual cuts split to 0, 1, ..., 19
    cuts = np.arange(20, dtype="float64")
    labtrans = LabTransform(cuts)

    x_train = get_x(deephit_train)
    x_val = get_x(deephit_val)
    x_test = get_x(deephit_test)

    y_train = labtrans.transform(*get_target(deephit_train))
    y_val = labtrans.transform(*get_target(deephit_val))
    y_test = labtrans.transform(*get_target(deephit_test))

    deephit_train = (x_train, y_train)
    deephit_val = (x_val, y_val)
    deephit_test = (x_test, y_test)

    if OPTN:
        return (deephit_train, deephit_test)
    else:
        return (deephit_train, deephit_val, deephit_test)


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


def load_optn_eval_deephit(variables :str, OPTN: int):

    graft_val, graft_test = _load_optn_eval(OPTN, "dynamic", "graft")
    graft_val_sta, graft_test_sta = _load_optn_eval(OPTN, "static", "graft")

    if variables == "full":
        pass
    elif variables == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]
    else:
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]

    deephit_val = process_iid(graft_val, graft_val_sta)
    deephit_test = process_iid(graft_test, graft_test_sta)

    get_x = lambda df: (df
                    .drop(columns=['T', 'E'])
                    .values.astype('float32'))
    get_target = lambda df: (df['T'].values, df['E'].values)
    # manual cuts split to 0, 1, ..., 19
    cuts = np.arange(20, dtype="float64")
    labtrans = LabTransform(cuts)

    x_val = get_x(deephit_val)
    x_test = get_x(deephit_test)

    y_val = labtrans.transform(*get_target(deephit_val))
    y_test = labtrans.transform(*get_target(deephit_test))

    deephit_val = (x_val, y_val)
    deephit_test = (x_test, y_test)

    return (deephit_val, deephit_test)



def _load_optn_eval(OPTN: int, mode: str, outcome: str="graft"):
    # load val and test portion of each OPTN region

    # for models trained on all data, when evaluating on OPTN regions,
    # we need to make sure that we are not evaluating its training set,
    # obtain val/test samples from that region.
    _, val, test = load_dynamic_cox(outcome, "ff")
    val_ids, test_ids = val.PX_ID.unique(), test.PX_ID.unique()

    optn_train, optn_val = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
    optn_train_sta, optn_val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")

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