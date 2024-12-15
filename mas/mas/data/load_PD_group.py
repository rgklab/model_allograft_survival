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


def load_PD_dataset(PD: str, outcome: str, mode: str):
    train, val, test = load_dynamic_cox(outcome, "ff") if mode == "dynamic" else load_SRTR_static_df(outcome)

    dataset =pd.concat([train, val, test])

    px_ids = {}
    for split in ["train", "val", "test"]:
        px_ids[split] = np.loadtxt(f"{SPLIT_DIR}/PD/{PD}/{split}_split.txt", delimiter='\n')

    train = dataset[dataset.PX_ID.isin(px_ids["train"])]
    val = dataset[dataset.PX_ID.isin(px_ids["val"])]
    test = dataset[dataset.PX_ID.isin(px_ids["test"])]

    return train, val, test


def create_PD_dataset():
    """
    create datasets that are startified by the primary diagnosis of the patients
    """
    tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    train_sta, val_sta, test_sta = load_SRTR_static_df("graft")
    dataset_sta =pd.concat([train_sta,val_sta,test_sta])

    ncc_ids, neo_ids, cld_ids, ahn_ids, ba_ids, mld_ids = _get_PD_ids(tx_li=tx_li)
    list = [ncc_ids, neo_ids, cld_ids, ahn_ids, ba_ids, mld_ids]

    censored = dataset_sta[dataset_sta.EVENT == 0]
    censored_ids = censored.PX_ID.unique()

    list = [ncc_ids, neo_ids, cld_ids, ahn_ids, ba_ids, mld_ids]
    strs = ["ncc", "neo", "cld", "ahn", "ba", "mld"]

    for ids, diag in tqdm(zip(list, strs), total=6):
        event_ids = dataset_sta[(dataset_sta.EVENT == 1) & (dataset_sta.PX_ID.isin(ids))].PX_ID.unique()
        total_ids = np.concatenate((event_ids,censored_ids))

        # We perform a 70%-15%-15% train-val-test split.
        train_splt, val_splt = 0.7, 0.85

        np.random.seed(3077)

        np.random.shuffle(total_ids)
        train, val, test = np.split(total_ids,
                        [int(train_splt*len(total_ids)), 
                        int(val_splt*len(total_ids))]
                        )

        with open(f"mas/data/data_splits/PD/{diag}/train_split.txt", "w") as f:
            f.write("\n".join(train.astype('str')))
        with open(f"mas/data/data_splits/PD/{diag}/val_split.txt", "w") as f:
            f.write("\n".join(val.astype('str')))
        with open(f"mas/data/data_splits/PD/{diag}/test_split.txt", "w") as f:
            f.write("\n".join(test.astype('str')))


def _get_PD_ids(tx_li: pd.DataFrame):
    tx_li = _find_PD(tx_li=tx_li)
    ncc_ids = tx_li[tx_li.PD == "Non Cholestatic Cirrhosis"].PX_ID
    neo_ids = tx_li[tx_li.PD == "Neoplasms (malignant or benign)"].PX_ID
    cld_ids = tx_li[tx_li.PD == "Cholestatic Liver Disease or Cirrhosis"].PX_ID
    ahn_ids = tx_li[tx_li.PD == "Acute Hepatic Necrosis"].PX_ID
    ba_ids = tx_li[tx_li.PD == "Biliary Atresia or Other Neonatal Disease"].PX_ID
    mld_ids = tx_li[tx_li.PD == "Metaboloic Liver Disease (hereditary / genetic)"].PX_ID

    return (ncc_ids, neo_ids, cld_ids, ahn_ids, ba_ids, mld_ids)



def _find_PD(tx_li: pd.DataFrame):
     ## Other
    tx_li.loc[((tx_li["CAN_DGN"] == 4290) |
            (tx_li["CAN_DGN"] == 4500) |
            (tx_li["CAN_DGN"] == 4510) |
            (tx_li["CAN_DGN"] == 4520) |
            (tx_li["CAN_DGN"] == 4597) |
            (tx_li["CAN_DGN"] == 4598)), "PD"] = "Other"

    # Acute Hepatic Necrosis
    tx_li.loc[((tx_li["CAN_DGN"] == 4100) |
            (tx_li["CAN_DGN"] == 4101) |
            (tx_li["CAN_DGN"] == 4102) |
            (tx_li["CAN_DGN"] == 4103) |
            (tx_li["CAN_DGN"] == 4104) |
            (tx_li["CAN_DGN"] == 4105) |
            (tx_li["CAN_DGN"] == 4106) |
            (tx_li["CAN_DGN"] == 4107) |
            (tx_li["CAN_DGN"] == 4108) |
            (tx_li["CAN_DGN"] == 4110)), "PD"] = "Acute Hepatic Necrosis"

    # Non Cholestatic Cirrhosis
    tx_li.loc[((tx_li["CAN_DGN"] == 4200) |
            (tx_li["CAN_DGN"] == 4201) |
            (tx_li["CAN_DGN"] == 4202) |
            (tx_li["CAN_DGN"] == 4203) |
            (tx_li["CAN_DGN"] == 4204) |
            (tx_li["CAN_DGN"] == 4205) |
            (tx_li["CAN_DGN"] == 4206) |
            (tx_li["CAN_DGN"] == 4207) |
            (tx_li["CAN_DGN"] == 4208) |
            (tx_li["CAN_DGN"] == 4209) |
            (tx_li["CAN_DGN"] == 4210) |
            (tx_li["CAN_DGN"] == 4212) |
            (tx_li["CAN_DGN"] == 4213) |
            (tx_li["CAN_DGN"] == 4214) |
            (tx_li["CAN_DGN"] == 4215) |
            (tx_li["CAN_DGN"] == 4216) |
            (tx_li["CAN_DGN"] == 4217) |
            (tx_li["CAN_DGN"] == 4592) |
            (tx_li["CAN_DGN"] == 4593)), "PD"] = "Non Cholestatic Cirrhosis"

    # Cholestatic Liver Disease or Cirrhosis
    tx_li.loc[((tx_li["CAN_DGN"] == 4220) |
            (tx_li["CAN_DGN"] == 4230) |
            (tx_li["CAN_DGN"] == 4231) |
            (tx_li["CAN_DGN"] == 4235) |
            (tx_li["CAN_DGN"] == 4240) |
            (tx_li["CAN_DGN"] == 4241) |
            (tx_li["CAN_DGN"] == 4242) |
            (tx_li["CAN_DGN"] == 4245) |
            (tx_li["CAN_DGN"] == 4250) |
            (tx_li["CAN_DGN"] == 4255) |
            (tx_li["CAN_DGN"] == 4260)), "PD"] = "Cholestatic Liver Disease or Cirrhosis"

    # Biliary Atresia or Other Neonatal Disease
    tx_li.loc[((tx_li["CAN_DGN"] == 4264) |
            (tx_li["CAN_DGN"] == 4265) |
            (tx_li["CAN_DGN"] == 4270) |
            (tx_li["CAN_DGN"] == 4271) |
            (tx_li["CAN_DGN"] == 4272) |
            (tx_li["CAN_DGN"] == 4275) |
            (tx_li["CAN_DGN"] == 4280) |
            (tx_li["CAN_DGN"] == 4285)), "PD"] = "Biliary Atresia or Other Neonatal Disease"

    # Metaboloic Liver Disease (hereditary / genetic)
    tx_li.loc[((tx_li["CAN_DGN"] == 4300) |
            (tx_li["CAN_DGN"] == 4301) |
            (tx_li["CAN_DGN"] == 4302) |
            (tx_li["CAN_DGN"] == 4303) |
            (tx_li["CAN_DGN"] == 4304) |
            (tx_li["CAN_DGN"] == 4305) |
            (tx_li["CAN_DGN"] == 4306) |
            (tx_li["CAN_DGN"] == 4307) |
            (tx_li["CAN_DGN"] == 4308) |
            (tx_li["CAN_DGN"] == 4315)), "PD"] = "Metaboloic Liver Disease (hereditary / genetic)"
            
    # Neoplasms (malignant or benign)
    tx_li.loc[((tx_li["CAN_DGN"] == 4400) |
            (tx_li["CAN_DGN"] == 4401) |
            (tx_li["CAN_DGN"] == 4402) |
            (tx_li["CAN_DGN"] == 4403) |
            (tx_li["CAN_DGN"] == 4404) |
            (tx_li["CAN_DGN"] == 4405) |
            (tx_li["CAN_DGN"] == 4410) |
            (tx_li["CAN_DGN"] == 4420) |
            (tx_li["CAN_DGN"] == 4430) |
            (tx_li["CAN_DGN"] == 4450) |
            (tx_li["CAN_DGN"] == 4451) |
            (tx_li["CAN_DGN"] == 4455)), "PD"] = "Neoplasms (malignant or benign)"

    return tx_li


def create_PD_mapping():
    """
    Maps patients to primary diagnosis.
    """
    tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    tx_li = tx_li[["PX_ID", "CAN_DGN"]]
    tx_li = _find_PD(tx_li)
    dump(tx_li[["PX_ID", "PD"]], "mas/data/reference_tables/PD.csv")



if __name__ == "__main__":
    import fire

    fire.Fire()