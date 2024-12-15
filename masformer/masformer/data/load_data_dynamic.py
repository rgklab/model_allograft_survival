import numpy as np
import pandas as pd
from tqdm import tqdm
import miceforest as mf

DATA_DIR = ""

def load_SRTR_dynamic(
    imputation : str,
    load_from_pkl : bool = True
    ):
    """
    Loads and interpolates dynamic time series data.

    """
    if load_from_pkl:
        train = pd.read_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/train_dynamic.pkl")
        val = pd.read_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/val_dynamic.pkl")
        test = pd.read_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/test_dynamic.pkl")
        
        return (train, val, test)
        
    else:
        # Load the TXF_LI and FOL_IMMUNO table into memory.
        txf_li = pd.read_sas(DATA_DIR + "txf_li.sas7bdat")

        txf_li["TIME_SINCE_BASELINE"] = (txf_li["TFL_PX_STAT_DT"] - txf_li["REC_TX_DT"]).dt.days/365

        # Load the patient IDs associated with the train, validation, and test 
        # splits into memory.
        px_ids = {}
        for split in ["train", "val", "test"]:
            with open(f"masformer/data/data_splits/{split}_split.txt") as f:
                px_ids[split] = [float(id) for id in f]
        
        # Extract the specified feature set from the loaded DataFrame.
        extractor = DynamicFeatureExtractor()
        txf_li = extractor.extract_features(txf_li)

        # Split into train, validation, and test sets.
        txf_li = txf_li[txf_li['PX_ID'].isin(px_ids["train"]+px_ids["val"]+px_ids["test"])]

        processor = ImputeProcessor()
        txf_li = eval(f"processor.{imputation}(txf_li)")

        train = txf_li[txf_li['PX_ID'].isin(px_ids["train"])]
        val = txf_li[txf_li['PX_ID'].isin(px_ids["val"])]
        test = txf_li[txf_li['PX_ID'].isin(px_ids["test"])]

        train.to_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/train_dynamic.pkl")
        val.to_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/val_dynamic.pkl")
        test.to_pickle(f"masformer/data/dynamic_postprocessed/{imputation}/test_dynamic.pkl")

        return (train, val, test)


class DynamicFeatureExtractor():
    
    def extract_features(self, df):
        df = df[~df["TIME_SINCE_BASELINE"].isnull()]

        processor = FeatureProcessor()

        out = pd.DataFrame(df["PX_ID"])

        # Add in the covariate features
        out = pd.concat([out, df["TIME_SINCE_BASELINE"]], axis=1)
        out = pd.concat([out, df["TFL_CREAT"]], axis=1)
        out = pd.concat([out, df["TFL_INR"]], axis=1)
        out = pd.concat([out, processor.yesnounknown_to_numeric(df["TFL_HOSP"])], axis=1)
        out = pd.concat([out, processor.yesnounknown_to_numeric(df["TFL_DIAB_DURING_FOL"])], axis=1)
        out = pd.concat([out, 
                pd.get_dummies(df["TFL_PRIMARY_PAY"], prefix="TFL_PRIMARY_PAY")],axis=1)
        out = pd.concat([out, df["TFL_TOT_BILI"]], axis=1)
        out = pd.concat([out, df["TFL_SGPT"]], axis=1)
        out = pd.concat([out, df["TFL_ALBUMIN"]], axis=1)
        out = pd.concat([out, df["TFL_SGOT"]], axis=1)
        out = pd.concat([out, df["TFL_BMI"]], axis=1)
        out = pd.concat([out, df["TFL_WGT_KG"]], axis=1)

        return out


class ImputeProcessor():

    def _forward_fill(self, df, pxid):
        df = df[df["PX_ID"] == pxid]

        df = df.set_index('TIME_SINCE_BASELINE').sort_index().ffill().bfill().reset_index()
        # df = df.drop(["TIME_SINCE_BASELINE"], axis=1)

        return df

    def ff(self, df):
       
        small_dfs = []

        for px in tqdm(df["PX_ID"].unique()):
        # Collect all covariates for that patient
            px_df = self._forward_fill(df, px)

            # For any columns that are null - fill them with the global
            # dataset-wide mean value for that column.
            for col in px_df.columns:
                px_df[col].fillna(df[col].mean(), inplace=True)

            small_dfs.append(px_df)

        dynamic_df = pd.concat(small_dfs, ignore_index=True)

        return dynamic_df

    def mice(self, df):
        ids = df["PX_ID"]
        kernel = mf.ImputationKernel(
            df.drop(columns=["PX_ID"]),
            datasets=5,
            save_all_iterations=False,
            random_state=42
        )

        kernel.mice(iterations=5, verbose=True)
        completed = kernel.complete_data(dataset=0)
        completed["PX_ID"] = ids

        return completed


    def mice_gbdt(self, df):
        ids = df["PX_ID"]
        kernel = mf.ImputationKernel(
            df.drop(columns=["PX_ID"]),
            datasets=5,
            save_all_iterations=False,
            random_state=42
        )

        kernel.mice(iterations=1, boosting='gbdt', min_sum_hessian_in_leaf=0.01, verbose=True)

        optimal_parameters, losses = kernel.tune_parameters(
            dataset=0, optimization_steps=5, verbose=True
        )

        kernel.mice(iterations=2, verbose=True, variable_parameters=optimal_parameters)
        completed = kernel.complete_data(dataset=0)
        completed["PX_ID"] = ids

        return completed


    def mice_strata(self, df):

        ids = df["PX_ID"]
        df = df.drop(columns=["PX_ID"])

        t1 = df[(df["TIME_SINCE_BASELINE"] <= 0.7)]
        t2 = df[(df["TIME_SINCE_BASELINE"] > 0.7) & (df["TIME_SINCE_BASELINE"] <= 1.5)]
        t3 = df[(df["TIME_SINCE_BASELINE"] > 1.5) & (df["TIME_SINCE_BASELINE"] <= 2.5)]
        t4 = df[(df["TIME_SINCE_BASELINE"] > 2.5) & (df["TIME_SINCE_BASELINE"] <= 3.5)]
        t5 = df[(df["TIME_SINCE_BASELINE"] > 3.5) & (df["TIME_SINCE_BASELINE"] <= 4.5)]
        t6 = df[(df["TIME_SINCE_BASELINE"] > 4.5) & (df["TIME_SINCE_BASELINE"] <= 5.5)]
        t7 = df[(df["TIME_SINCE_BASELINE"] > 5.5) & (df["TIME_SINCE_BASELINE"] <= 6.5)]
        t8 = df[(df["TIME_SINCE_BASELINE"] > 6.5) & (df["TIME_SINCE_BASELINE"] <= 7.5)]
        t9 = df[(df["TIME_SINCE_BASELINE"] > 7.5) & (df["TIME_SINCE_BASELINE"] <= 8.5)]
        t10 = df[(df["TIME_SINCE_BASELINE"] > 8.5) & (df["TIME_SINCE_BASELINE"] <= 9.5)]
        t11 = df[(df["TIME_SINCE_BASELINE"] > 9.5) & (df["TIME_SINCE_BASELINE"] <= 10.5)]
        t12 = df[(df["TIME_SINCE_BASELINE"] > 10.5) & (df["TIME_SINCE_BASELINE"] <= 11.5)]
        t13 = df[(df["TIME_SINCE_BASELINE"] > 11.5)]

        df_list = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]
        imputed = []

        for i, dataset in enumerate(tqdm(df_list)):
            if i <= 2:
                kernel = mf.ImputationKernel(
                    dataset,
                    datasets=5,
                    save_all_iterations=False,
                    random_state=42
                )
                kernel.mice(5, verbose=True)
                imputed.append(kernel.complete_data(dataset=0))
            else:
                df = pd.concat([dataset, imputed[-1], imputed[-2]])
                kernel = mf.ImputationKernel(
                    df,
                    datasets=5,
                    save_all_iterations=False,
                    random_state=42
                )
                kernel.mice(5, verbose=True)
                imputed.append(kernel.complete_data(dataset=0).iloc[:dataset.shape[0],:])

        imputed_df = pd.concat(imputed).sort_index()
        
        imputed_df["PX_ID"] = ids

        return imputed_df


class FeatureProcessor():
    """
    Utility class to gather together feature processing functions.
    """
    def __init__(self):
        MELD_CONVERSION_DIR = "masformer/data/reference_tables/meld_conversion.csv"
        # MELD_CONVERSION_DIR = "reference_tables/meld_conversion.csv"
        meld_conversion_cand_liin = pd.read_csv(MELD_CONVERSION_DIR, header=None)
        self.meld_conversion = dict(meld_conversion_cand_liin.values)

        FUNCSTAT_CONVERSION_DIR = "masformer/data/reference_tables/funcstat_conversion.csv"
        # FUNCSTAT_CONVERSION_DIR = "reference_tables/funcstat_conversion.csv"
        funcstat_conversion_cand_liin = pd.read_csv(FUNCSTAT_CONVERSION_DIR, header=None)
        self.funcstat_conversion = dict(funcstat_conversion_cand_liin.values)

    def convert_MELD(self, col):
        return col.apply(lambda x : self.meld_conversion[x])

    def convert_FUNCSTAT(self, col):
        def safe_mapping(key):
            try:
                return self.funcstat_conversion[key]
            except KeyError:
                return key
        col = col.apply(lambda x : safe_mapping(x))
        return col.fillna(col.mean())

    def transplant_ctr_to_optn_region(self, col):
        """
        Maps transplant center ID to OPTN region.
        """
        institutions = pd.read_sas(DATA_DIR + "institution.sas7bdat")
        institutions_to_regions = {
            ins:reg for ins, reg in zip(institutions["CTR_ID"], institutions["REGION"])
        }
        def safe_mapping(key):
            try:
                return institutions_to_regions[key]
            except KeyError:
                return -1
        return col.apply(safe_mapping)

    def yesnounknown_to_numeric(self, col):
        mapping = {b'N': 0, b'Y': 1, b'U': np.nan}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return np.nan
        return col.apply(safe_mapping)

    def ordinal_meanpadding(self, col, exceptions):
        def apply_ordinal(elem):
            if elem in exceptions:
                return np.nan
            return elem
        return col.apply(apply_ordinal).fillna(col.mean())

    def ethnicity_binary(self, col):
        return col.apply(lambda x : 1 if x == b'LATINO' else 0)

    def gender_binary(self, col):
        return col.apply(lambda x : 1 if x == b'F' else 0)

    def don_type_binary(self, col):
        return col.apply(lambda x : 1 if x == b'L' else 0)

if __name__ == "__main__":
    load_SRTR_dynamic(imputation="ff", load_from_pkl=False)
    # load_SRTR_dynamic(imputation="mice", load_from_pkl=False)
    # load_SRTR_dynamic(imputation="mice_strata", load_from_pkl=False)
    # load_SRTR_dynamic(imputation="mice_gbdt", load_from_pkl=False)
