import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_data_dynamic import load_SRTR_dynamic
from mas.data.load_dcox_delta import load_dcox_delta
from mas.data.load_data_dynamic import load_SRTR_dynamic
from mas.data.load_data import load_SRTR_static, load_SRTR_static_df
from joblib import load, dump

DATA_DIR = ""
SPLIT_DIR = ""
variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]


def _find_GFC(tx_li: pd.DataFrame):
     
    tx_li.loc[tx_li.REC_FAIL_HEP_DENOVO == b'Y', 
                    "GF_causes"] = "Hepatitis: DeNovo"

    tx_li.loc[tx_li.REC_FAIL_HEP_RECUR == b'Y', 
                    "GF_causes"] = "Hepatitis: Recurrent"

    tx_li.loc[tx_li.REC_FAIL_INFECT == b'Y', 
                    "GF_causes"] = "Infection"

    tx_li.loc[tx_li.REC_FAIL_PRIME_GRAFT_FAIL == b'Y', 
                    "GF_causes"] = "Primary Graft Failure"

    tx_li.loc[tx_li.REC_FAIL_RECUR_DISEASE == b'Y', 
                    "GF_causes"] = "Recurrent Disease"

    tx_li.loc[tx_li.REC_FAIL_REJ_ACUTE == b'Y', 
                    "GF_causes"] = "Acute Rejection"

    tx_li.loc[tx_li.REC_FAIL_VASC_THROMB == b'Y', 
                    "GF_causes"] = "Vascular Thrombosis"

    return tx_li


def create_PD_mapping():
    """
    Maps patients to primary diagnosis.
    """
    tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    tx_li = _find_GFC(tx_li)
    dump(tx_li[["PX_ID", "GF_causes"]], "mas/data/reference_tables/GFC.csv")



if __name__ == "__main__":
    import fire

    fire.Fire()