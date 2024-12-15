import pandas as pd
import numpy as np
from tqdm import tqdm

from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_data import load_SRTR_static_df
from joblib import load, dump

DATA_DIR = ""
SPLIT_DIR = ""
variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]


def load_OPTN_dataset(OPTN: int, outcome: str, mode: str, normalized=True):
    train, val, test = load_dynamic_cox(outcome, "ff", normalized=normalized) if mode == "dynamic" else load_SRTR_static_df(outcome)
    dataset =pd.concat([train, val, test])

    px_ids = {}
    for split in ["train", "test"]:
        px_ids[split] = np.loadtxt(f"{SPLIT_DIR}/OPTN/region_{OPTN}/{split}_split.txt", delimiter='\n')

    train = dataset[dataset.PX_ID.isin(px_ids["train"])]
    test = dataset[dataset.PX_ID.isin(px_ids["test"])]

    return train, test


def create_OPTN_dataset():
    """
    create datasets that are startified by the OPTN regions. 
    """
    optn = load("mas/data/reference_tables/optn.csv")
    train_sta, val_sta, test_sta = load_SRTR_static_df("graft")
    dataset_sta =pd.concat([train_sta,val_sta,test_sta])
    dataset_sta = dataset_sta.merge(optn, how="left", on="PX_ID")

    for i in tqdm(range(1, 12)):
        ids = dataset_sta[dataset_sta.OPTN == i].PX_ID.unique()
        train_splt, test_splt = 0.55, 0.45

        np.random.seed(3077)
        np.random.shuffle(ids)

        train, test = np.split(ids,
                        [int(train_splt*len(ids))]
                        )

        with open(f"mas/data/data_splits/OPTN/region_{i}/train_split.txt", "w") as f:
            f.write("\n".join(train.astype('str')))
        with open(f"mas/data/data_splits/OPTN/region_{i}/test_split.txt", "w") as f:
            f.write("\n".join(test.astype('str'))) 


def create_OPTN_mapping():
    """
    Maps patients to OPTN region.
    """
    tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    tx_li["OPTN"] = _transplant_ctr_to_optn_region(tx_li["REC_CTR_ID"])
    dump(tx_li[["PX_ID", "OPTN"]], "mas/data/reference_tables/optn.csv")


def _transplant_ctr_to_optn_region(col):
    """
    Maps transplant center ID to OPTN region.
    """
    institutions = pd.read_csv(f"mas/data/reference_tables/institution.csv")
    institutions_to_regions = {
        ins:reg for ins, reg in zip(institutions["CTR_ID"], institutions["REGION"])
    }
    def safe_mapping(key):
        try:
            return institutions_to_regions[key]
        except KeyError:
            return -1
    return col.apply(safe_mapping)


if __name__ == "__main__":
    import fire

    fire.Fire()