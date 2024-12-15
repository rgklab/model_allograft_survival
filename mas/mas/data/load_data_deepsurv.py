import numpy as np
import pandas as pd

from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_data import load_SRTR_static, load_SRTR_static_df
from mas.data.load_PD_group import load_PD_dataset
from mas.data.load_OPTN import load_OPTN_dataset
from utils import load_optn_eval


def load_deepsurv(outcome, variables, normalized = True, PD = None, OPTN = None):
    if PD:
        train, val, test = load_PD_dataset(PD, outcome, "dynamic")
    elif OPTN:
        train, val, test = load_dynamic_cox(outcome, "ff", normalized=normalized)
        train, test = load_OPTN_dataset(OPTN, outcome, "dynamic")
    else:
        train, val, test = load_dynamic_cox(outcome, "ff", normalized=normalized)

    # if delta:
    #     var_deltas = [i+"_delta" for i in variables]
    #     variables = variables + var_deltas

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

    if PD:
        train_sta, val_sta, test_sta = load_PD_dataset(PD, outcome, "static")
    elif OPTN:
        train_sta, val_sta, test_sta = load_SRTR_static_df(outcome)
        train_sta, test_sta = load_OPTN_dataset(OPTN, outcome, "static")
    else:
        train_sta, val_sta, test_sta = load_SRTR_static_df(outcome)

    deepsurv_train = _process_iid(train, train_sta)
    deepsurv_train = _format_deepsurv(deepsurv_train)

    deepsurv_val = _process_iid(val, val_sta)
    deepsurv_val = _format_deepsurv(deepsurv_val)

    deepsurv_test = _process_iid(test, test_sta)
    deepsurv_test = _format_deepsurv(deepsurv_test)

    if OPTN:
        return (deepsurv_train, deepsurv_test)
    else:
        return (deepsurv_train, deepsurv_val, deepsurv_test)


def load_sksurv(outcome, variables, normalized = True, PD = None, OPTN = None):
    if PD:
        train, val, test = load_PD_dataset(PD, outcome, "dynamic")
    elif OPTN:
        train, val, test = load_dynamic_cox(outcome, "ff", normalized=normalized)
        train, test = load_OPTN_dataset(OPTN, outcome, "dynamic")
    else:
        train, val, test = load_dynamic_cox(outcome, "ff", normalized=normalized)

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

    if PD:
        train_sta, val_sta, test_sta = load_PD_dataset(PD, outcome, "static")
    elif OPTN:
        train_sta, val_sta, test_sta = load_SRTR_static_df(outcome)
        train_sta, test_sta = load_OPTN_dataset(OPTN, outcome, "static")
    else:
        train_sta, val_sta, test_sta = load_SRTR_static_df(outcome)

    sksurv_train = _process_iid(train, train_sta)
    sksurv_train = _format_sksurv(sksurv_train)

    sksurv_val = _process_iid(val, val_sta)
    sksurv_val = _format_sksurv(sksurv_val)

    sksurv_test = _process_iid(test, test_sta)
    sksurv_test = _format_sksurv(sksurv_test)

    if OPTN:
        return(sksurv_train, sksurv_test)
    else:
        return (sksurv_train, sksurv_val, sksurv_test)


def load_optn_eval_deepsurv(outcome, variables, OPTN):
    """
    load dataset for optn evaluations of model trained on all SRTR
    """

    graft_val, graft_test = load_optn_eval(OPTN, "dynamic", outcome)
    graft_val_sta, graft_test_sta = load_optn_eval(OPTN, "static", outcome)
    
    
    if variables == "full":
        pass
    elif variables == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]
    else:
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]

    deepsurv_val = _process_iid(graft_val, graft_val_sta)
    deepsurv_val = _format_deepsurv(deepsurv_val)

    deepsurv_test = _process_iid(graft_test, graft_test_sta)
    deepsurv_test = _format_deepsurv(deepsurv_test)


    return (deepsurv_val, deepsurv_test)


def load_optn_eval_sksurv(outcome, variables, OPTN):
    """
    load dataset for optn evaluations of model trained on all SRTR
    """

    graft_val, graft_test = load_optn_eval(OPTN, "dynamic", outcome)
    graft_val_sta, graft_test_sta = load_optn_eval(OPTN, "static", outcome)
    
    
    if variables == "full":
        pass
    elif variables == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]
    else:
        graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
        graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]

    # sksurv_val = _process_iid(graft_val, graft_val_sta)
    # sksurv_val = _format_sksurv(sksurv_val)

    # sksurv_test = _process_iid(graft_test, graft_test_sta)
    # sksurv_test = _format_sksurv(sksurv_test)

    sksurv = _process_iid(pd.concat([graft_val,graft_test]), pd.concat([graft_val_sta, graft_test_sta]))
    sksurv = _format_sksurv(sksurv)


    return sksurv



def _process_iid(df, static_df):
    iid_df = df.drop(columns=["EVENT", "start"]).merge(
        static_df[["PX_ID", "EVENT", "TIME"]], on="PX_ID", how="left")

    iid_df["TIME"] = iid_df["TIME"] - iid_df["stop"]
    iid_df.rename(columns={"TIME":"T", "EVENT":"E"}, inplace=True)
    return iid_df.drop(columns=["PX_ID", "stop"])


def _format_deepsurv(df):
    T = np.array(df["T"])
    E = np.array(df["E"])
    X = np.array(df.drop(["T", "E"], axis=1))
    return {
        'x' : X.astype(np.float32),
        't' : T.astype(np.float32),
        'e' : E.astype(np.int32)
    }


def _format_sksurv(df):
    T = np.array(df["T"])
    E = np.array(df["E"])
    X = np.array(df.drop(["T", "E"], axis=1))
    Y = np.array([(bool(e), t) for (e, t) in zip(E, T)], dtype=[("e", bool), ("t", float)])
    return (X, Y)
    