import numpy as np
import pandas as pd

from mas.data.load_dynamic_cox import load_dynamic_cox


# def load_optn_eval(OPTN: int, mode: str, outcome: str="graft", deephit: bool=False):
#     # load val and test portion of each OPTN region

#     # for models trained on all data, when evaluating on OPTN regions,
#     # we need to make sure that we are not evaluating its training set,
#     # obtain val/test samples from that region.
#     _, val, test = load_dynamic_cox(outcome, "ff")
#     val_ids, test_ids = val.PX_ID.unique(), test.PX_ID.unique()

#     if deephit:
#         optn_train, optn_val = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
#         optn_train_sta, optn_val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
#     else:
#         optn_train, optn_val = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="dynamic")
#         optn_train_sta, optn_val_sta = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="static")

#     optn_data, optn_data_sta = pd.concat([optn_train, optn_val]), pd.concat([optn_train_sta, optn_val_sta])

#     graft_val, graft_test = optn_data[optn_data.PX_ID.isin(val_ids)], optn_data[optn_data.PX_ID.isin(test_ids)]
#     graft_val_sta = optn_data_sta[optn_data_sta.PX_ID.isin(val_ids)]
#     graft_test_sta = optn_data_sta[optn_data_sta.PX_ID.isin(test_ids)]

#     if mode == "dynamic":
#         return graft_val, graft_test
#     else:
#         return graft_val_sta, graft_test_sta



def get_IDs():
    """
    obtain PX_IDs that exist in the dcox dataset, the true set of patients we are using
    """
    train,val,test = load_dynamic_cox("graft", "ff")
    dataset = pd.concat([train,val,test])
    ids = dataset.PX_ID.unique()
    return ids