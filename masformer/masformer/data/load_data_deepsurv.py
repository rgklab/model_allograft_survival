import numpy as np

from masformer.data.load_dynamic_cox import load_dynamic_cox
from masformer.data.load_data import load_SRTR_static, load_SRTR_static_df


def load_deepsurv(outcome, variables, delta : bool = False):
    train, val, test = load_dynamic_cox(outcome, delta=delta)

    if delta:
        var_deltas = [i+"_delta" for i in variables]
        variables = variables + var_deltas

    train = train[["PX_ID", "EVENT", "start", "stop"] + variables]
    val = val[["PX_ID", "EVENT", "start", "stop"] + variables]
    test = test[["PX_ID", "EVENT", "start", "stop"] + variables]

    train_sta, val_sta, test_sta = load_SRTR_static_df(outcome)

    deepsurv_train = _process_iid(train, train_sta)
    deepsurv_train = _format_deepsurv(deepsurv_train)

    deepsurv_val = _process_iid(val, val_sta)
    deepsurv_val = _format_deepsurv(deepsurv_val)

    deepsurv_test = _process_iid(test, test_sta)
    deepsurv_test = _format_deepsurv(deepsurv_test)

    return (deepsurv_train, deepsurv_val, deepsurv_test)


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
    