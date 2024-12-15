import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, Tuple, Dict

DATA_DIR = ""
NUM_YEARS = 100

def load_SRTR_static_df(
    outcome : str, 
    load_from_pkl : bool = True,
    tx_li : pd.DataFrame = None,
    txf_li: pd.DataFrame = None,
    feature: str = "all"
    ):

    if load_from_pkl:
        train = pd.read_pickle(f"mas/data/static_postprocessed/train_static_{outcome}_{feature}.pkl")
        val = pd.read_pickle(f"mas/data/static_postprocessed/val_static_{outcome}_{feature}.pkl")
        test = pd.read_pickle(f"mas/data/static_postprocessed/test_static_{outcome}_{feature}.pkl")

        return (train, val, test)
        
    else:
        statics = load_SRTR_static(outcome=outcome, format="dynamic", tx_li=tx_li, txf_li=txf_li, feature=feature)
        train_static, val_static, test_static = {}, {}, {}

        col_names = statics[3]
        ids = statics[4]
        num_col = statics[0]['x'].shape[1]

        for i in tqdm(range(num_col), desc="decode static SRTR"):
            train_static[col_names[i]] = statics[0]['x'][:,i]
            val_static[col_names[i]] = statics[1]['x'][:,i]
            test_static[col_names[i]] = statics[2]['x'][:,i]

        train_static["TIME"] = statics[0]['t']
        val_static["TIME"] = statics[1]['t']
        test_static["TIME"] = statics[2]['t']

        train_static["EVENT"] = statics[0]['e']
        val_static["EVENT"] = statics[1]['e']
        test_static["EVENT"] = statics[2]['e']

        train_static["PX_ID"] = ids['train']
        val_static["PX_ID"] = ids['val']
        test_static["PX_ID"] = ids['test']

        train_df_static = pd.DataFrame(train_static)
        val_df_static = pd.DataFrame(val_static)
        test_df_static = pd.DataFrame(test_static)

        train_df_static.to_pickle(f"mas/data/static_postprocessed/train_static_{outcome}_{feature}.pkl")
        val_df_static.to_pickle(f"mas/data/static_postprocessed/val_static_{outcome}_{feature}.pkl")
        test_df_static.to_pickle(f"mas/data/static_postprocessed/test_static_{outcome}_{feature}.pkl")


        return (train_df_static, val_df_static, test_df_static)



def load_SRTR_static(
    outcome : str, 
    format : str, 
    normalize : list = ["X"],
    tx_li : pd.DataFrame = None,
    txf_li: pd.DataFrame = None,
    return_col_names : bool = False,
    feature: str = "all"
    ) -> Union[
        Tuple[
            Dict[str, np.ndarray], 
            Dict[str, np.ndarray], 
            Dict[str, np.ndarray]
        ],
        Tuple[
            np.ndarray, np.ndarray
        ]
    ]:
    """
    Loads static data for each patient from the SRTR dataset into memory.

    Parameters
    ----------
    outcome : string
        The specific complication to extract from the table. Must be one of the
        following:
        - "infection" - infection as event.
        - "cancer" - cancer as event.
        - "graft" - graft failure as event.

    format : string
        The data format to return. Must be one of the following:
        - "scikit-survival" - each data split will be returned as a tuple of
            the form X, Y, where X is a NumPy array of shape 
            (n_samples, n_features) containing patient covariates, and Y is a 
            structured array of shape (n_samples,) containing the binary event 
            indicator as the first field, and time (of event/censoring) as the 
            second field. This is the format passed into the .fit function of 
            scikit-survival models.
        - "DeepSurv" - each data split will be returned as a dictionary, with
            the following key-value mappings:
            {
                'x' -> np.ndarray of shape (n_samples, n_features) containing
                    patient covariates (dtype of float32).
                't' -> np.ndarray of shape (n_samples,) containing the time of
                    event/censoring for each patient (dtype of float32).
                'e' -> np.ndarray of shape (n_samples,) containing binary
                    event indicators (dtype of int32).
            }

    tx_li : pd.DataFrame
        Optional argument. If provided, this argument is the raw TX_LI
        DataFrame which will be used instead of loading directly from memory.
        This argument is useful when training multiple model classes
        simultaneously: we can pre-load TX_LI into RAM once, and call this 
        function multiple times with the tx_li argument to format the data 
        as desired for each model class.

    txf_li : pd.DataFrame
        Optional argument. If provided, this argument is the raw TXF_LI
        DataFrame which will be used instead of loading directly from memory.
        This argument is useful when training multiple model classes
        simultaneously in the same way as the txf_Li argument.

    normalize : list
        List containing either or both "X", "T". If "X" is included, each
        feature (patient covariate) will be normalized; if "T" is included, each
        time-to-event data point will be normalized.

    feature : str
        Either "all" or "physiological"

    Returns
    -------
    train : one of tuple, dictionary.
        The training split of the data with the specified features.

    val : one of tuple, dictionary.
        The validation split of the data with the specified features.

    test : one of tuple, dictionary.
        The test split of the data with the specified features.
    """
    if tx_li is None:
        # Load the TX_LI table into memory.
        tx_li = pd.read_sas(DATA_DIR + "tx_li.sas7bdat")
    
    if txf_li is None:
        txf_li = pd.read_sas(DATA_DIR + "txf_li.sas7bdat")

    txf_li["time_after_transplant"] = (txf_li["TFL_PX_STAT_DT"] - txf_li["REC_TX_DT"]).dt.days

    # Load the patient IDs associated with the train, validation, and test 
    # splits into memory.
    px_ids = {}
    for split in ["train", "val", "test"]:
        px_ids[split] = np.loadtxt(f"mas/data/data_splits/{split}_split.txt", delimiter='\n')
        # px_ids[split] = np.loadtxt(f"data_splits/{split}_split.txt", delimiter='\n')
    # Extract the specified feature set from the loaded DataFrame.
    extractor = XTExtractor()

    tx_li = tx_li[tx_li['PX_ID'].isin(np.append(np.append(px_ids['train'],px_ids['val']), px_ids['test']))]

    X = extractor.extract_features(tx_li, feature)
    # Remove any columns that will become NaN upon normalization. Even when not
    # normalized, these columns do not have sufficient variation for a model to
    # meaningfully learn from, so we drop them.
    validator = DatasetValidator()
    X = validator.validate_dataframe(X, px_ids)

    # Split into train, validation, and test sets.
    train_set = X[X['PX_ID'].isin(px_ids["train"])]
    val_set = X[X['PX_ID'].isin(px_ids["val"])]
    test_set = X[X['PX_ID'].isin(px_ids["test"])]

    train_tx_li = tx_li[tx_li['PX_ID'].isin(px_ids["train"])]
    val_tx_li = tx_li[tx_li['PX_ID'].isin(px_ids["val"])]
    test_tx_li = tx_li[tx_li['PX_ID'].isin(px_ids["test"])]

    if outcome == "infection":
        mapping, col_names, ids = extractor.extract_infection_XTE([train_set, val_set, test_set], txf_li, NUM_YEARS)
        train_XTE = (mapping["train"][0], mapping["train"][1], mapping["train"][2])
        val_XTE = (mapping["val"][0], mapping["val"][1], mapping["val"][2])
        test_XTE = (mapping["test"][0], mapping["test"][1], mapping["test"][2])
    elif outcome == "cancer":
        mapping, col_names, ids = extractor.extract_cancer_XTE([train_set, val_set, test_set], txf_li, NUM_YEARS)
        train_XTE = (mapping["train"][0], mapping["train"][1], mapping["train"][2])
        val_XTE = (mapping["val"][0], mapping["val"][1], mapping["val"][2])
        test_XTE = (mapping["test"][0], mapping["test"][1], mapping["test"][2])
    elif outcome == "graft":
        mapping, col_names, ids = extractor.extract_graft_XTE([train_set, val_set, test_set], txf_li, NUM_YEARS)
        train_XTE = (mapping["train"][0], mapping["train"][1], mapping["train"][2])
        val_XTE = (mapping["val"][0], mapping["val"][1], mapping["val"][2])
        test_XTE = (mapping["test"][0], mapping["test"][1], mapping["test"][2])
    elif outcome == "mortality":
        mapping, col_names, ids = extractor.extract_mortality_XTE([train_set, val_set, test_set], 
                                        txf_li, NUM_YEARS, [train_tx_li, val_tx_li, test_tx_li])
        train_XTE = (mapping["train"][0], mapping["train"][1], mapping["train"][2])
        val_XTE = (mapping["val"][0], mapping["val"][1], mapping["val"][2])
        test_XTE = (mapping["test"][0], mapping["test"][1], mapping["test"][2])
    else:
        raise ValueError("Argument 'outcome' is not one of 'infection', \
            'cancer', 'infection'.")
    # Split the training, validation, and test sets each into X, T, E 
    # sub-arrays.
    formatter = DatasetFormatter()

    # If specified, normalize X and/or T.
    X_train, _, _ = train_XTE
    train_mean = np.average(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_XTE = formatter.normalize_XT(normalize, train_XTE, train_mean, train_std)
    val_XTE = formatter.normalize_XT(normalize, val_XTE, train_mean, train_std)
    test_XTE = formatter.normalize_XT(normalize, test_XTE, train_mean, train_std)
    
    # Return X, T, E in the specified format.
    if format == "scikit-survival":
        if return_col_names:
            return (
                formatter.format_scikit_survival(*train_XTE),
                formatter.format_scikit_survival(*val_XTE),
                formatter.format_scikit_survival(*test_XTE),
                list(col_names)
            )
        else:
            return (
                formatter.format_scikit_survival(*train_XTE),
                formatter.format_scikit_survival(*val_XTE),
                formatter.format_scikit_survival(*test_XTE)
            )
    elif format == "DeepSurv":
        if return_col_names:
            return (
                formatter.format_deepsurv(*train_XTE),
                formatter.format_deepsurv(*val_XTE),
                formatter.format_deepsurv(*test_XTE),
                list(col_names)
            )
        else:
            return (
                formatter.format_deepsurv(*train_XTE),
                formatter.format_deepsurv(*val_XTE),
                formatter.format_deepsurv(*test_XTE)  
            )
    elif format == "dynamic":
        return (
                formatter.format_dynamic(*train_XTE),
                formatter.format_dynamic(*val_XTE),
                formatter.format_dynamic(*test_XTE),
                list(col_names),
                ids
            )
    else:
        raise ValueError("Argument 'format' is not one of 'scikit-survival', \
            'DeepSurv'.")

class DatasetFormatter():

    def normalize_XT(
        self, 
        normalize : list, 
        XTE : Tuple[np.ndarray, np.ndarray, np.ndarray],
        mean = None,
        std = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Z-normalization on X, T, if specified in 'normalize'.

        Parameters
        ----------
        normalize : array-like
            Array-like; used to control if either/both of X and T are normalized
            by this function. If 'normalize' contains "X", this function returns
            a z-normalized X; if 'normalize' contains "T", this function 
            returns a z-normlaized T. Otherwise, un-normalized versions of X,
            T are returned.

        XTE : tuple of np.ndarrays
            Tuple of np.ndarrays corresponding to X (patient covariates), T
            (time-to-event data), and E (binary event indicators), in that
            order.

        Returns
        -------
        X : np.ndarray 
            np.ndarray of shape (n_samples, n_features) containing covariates
            for each patient; each feature is z-normalized if specified in the 
            `normalize` argument to this function.

        T : np.ndarray 
            np.ndarray of shape (n_samples) containing time-to-event data 
            (in days) for each patient; this array is z-normalized if specified 
            in the `normalize` argument.

        E : np.ndarray
            np.ndarray of shape (n_samples,) containing binary event indicators 
            for each patient. This array is NOT normalized; it is provided as
            a convenience for tuple packing/unpacking in the caller.
        """
        X, T, E = XTE

        if (mean is not None) and (std is not None):
            print("correct normalization")
            if "X" in normalize:
                X = (X - mean) / std
            return X, T, E
        else:
            if "X" in normalize:
                X = (X - np.average(X, axis=0)) / np.std(X, axis=0)
            if "T" in normalize:
                T = (T - np.average(T)) / np.std(T)
            return X, T, E

    def format_scikit_survival(
        self,
        X : np.ndarray,
        T : np.ndarray, 
        E : np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Formats given arrays X, T, E such that they can be directly passed into
        the .fit() function associated with a scikit-survival model.
        
        Parameters
        ----------
        X : np.ndarray 
            np.ndarray of shape (n_samples, n_features) containing covariates
            for each patient, excluding the death/censorship date and activation
            date.

        T : np.ndarray 
            np.ndarray of shape (n_samples) containing time-to-event data 
            (in days) for each patient.

        E : np.ndarray
            np.ndarray of shape (n_samples,) containing binary event indicators 
            for each patient.

        Returns
        -------
        X : np.ndarray
            np.ndarray of shape (n_samples, n_features) containing covariates.

        Y : np.ndarray
            np.ndarray containing time-to-event and binary event indicator data.
        """
        Y = np.array([(bool(e), t) for (e, t) in zip(E, T)], dtype=[("e", bool), ("t", float)])
        return X, Y

    def format_deepsurv(
        self,
        X : np.ndarray,
        T : np.ndarray, 
        E : np.ndarray
        ) -> Dict[str, np.ndarray]:
        """ 
        Formats given arrays X, T, E such that they can be directly passed into
        the .fit() function associated with a DeepSurv model.
        
        Parameters
        ----------
        X : np.ndarray 
            np.ndarray of shape (n_samples, n_features) containing covariates
            for each patient, excluding the death/censorship date and activation
            date.

        T : np.ndarray 
            np.ndarray of shape (n_samples) containing time-to-event data 
            (in days) for each patient.

        E : np.ndarray
            np.ndarray of shape (n_samples,) containing binary event indicators 
            for each patient.

        Returns
        -------
        dict : dict
            Dictionary containing X, T, and E arrays, in which the arrays are
            formatted to the correct numeric precision.
        """
        return {
            'x' : X.astype(np.float32),
            't' : T.astype(np.float32),
            'e' : E.astype(np.int32)
        }

    def format_dynamic(
        self,
        X : np.ndarray,
        T : np.ndarray, 
        E : np.ndarray
        ) -> Dict[str, np.ndarray]:
        """ 
        Formats given arrays X, T, E such that they can be directly passed into
        the .fit() function associated with a DeepSurv model.
        
        Parameters
        ----------
        X : np.ndarray 
            np.ndarray of shape (n_samples, n_features) containing covariates
            for each patient, excluding the death/censorship date and activation
            date.

        T : np.ndarray 
            np.ndarray of shape (n_samples) containing time-to-event data 
            (in days) for each patient.

        E : np.ndarray
            np.ndarray of shape (n_samples,) containing binary event indicators 
            for each patient.

        Returns
        -------
        dict : dict
            Dictionary containing X, T, and E arrays, in which the arrays are
            formatted to the correct numeric precision.
        """
        return {
            'x' : X,
            't' : T,
            'e' : E
        }

class XTExtractor():
    
    def extract_features(self, tx_li, feature):
        """ 
        Accepts as input a dataframe of the form in TX_LI. Drops columns
        containing information that is unavailable to physicians at the time of
        listing, and formats columns appropriately so that a learning model
        may be trained on them.
        """
        processor = FeatureProcessor()
        # We start with a new column composed of the PX_ID. Then, we append new
        # columns to this one, derived from the original dataframe. At the end,
        # we delete the PX_ID column (as it only serves as an indexer).
        # Drop the columns that provide too much information

        out = pd.DataFrame(tx_li["PX_ID"])
        # Then we append columns as needed

        if feature == "all":
            out = pd.concat([out, tx_li["CANHX_MPXCPT_HCC_APPROVE_IND"]], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_ANGINA"], prefix="CAN_ANGINA")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_ANGINA_CAD"], prefix="CAN_ANGINA_CAD")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_BACTERIA_PERIT"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_CEREB_VASC"])],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_DIAB_TY"].fillna(998.0), prefix="CAN_DIAB_TY")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_DRUG_TREAT_COPD"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_DRUG_TREAT_HYPERTEN"])],axis=1)
            out = pd.concat([out,
                processor.ordinal_meanpadding(tx_li["CAN_EDUCATION"], {996, 998})], axis=1)
            out = pd.concat([out, 
                processor.ethnicity_binary(tx_li["CAN_ETHNICITY_SRTR"])], axis=1)
            out = pd.concat([out, 
                processor.gender_binary(tx_li["CAN_GENDER"])], axis=1)
            out = pd.concat([out, tx_li["CAN_HGT_CM"].fillna(tx_li["CAN_HGT_CM"].mean())], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_ALBUMIN"].fillna(tx_li["CAN_LAST_ALBUMIN"].mean())], axis=1)
            out = pd.concat([out,
                processor.ordinal_meanpadding(tx_li["CAN_LAST_ASCITES"], {4})], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_BILI"].fillna(tx_li["CAN_LAST_BILI"].mean())], axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_LAST_DIAL_PRIOR_WEEK"])],axis=1)
            out = pd.concat([out,
                processor.ordinal_meanpadding(tx_li["CAN_LAST_ENCEPH"], {4})], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_INR"].fillna(tx_li["CAN_LAST_INR"].mean())], axis=1)
            out = pd.concat([out, 
                processor.convert_MELD(tx_li["CAN_LAST_SRTR_LAB_MELD"])], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_INR"].fillna(tx_li["CAN_LAST_INR"].mean())], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_SERUM_SODIUM"].fillna(tx_li["CAN_LAST_SERUM_SODIUM"].mean())], axis=1)
            out = pd.concat([out,
                pd.get_dummies(tx_li["CAN_LAST_STAT"], prefix="CAN_LAST_STAT")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_MALIG"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PERIPH_VASC"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PORTAL_VEIN"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PREV_ABDOM_SURG"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PULM_EMBOL"])],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_RACE_SRTR"], prefix="CAN_RACE_SRTR")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_TIPSS"])],axis=1)
            out = pd.concat([out, tx_li["CAN_WGT_KG"].fillna(tx_li["CAN_WGT_KG"].mean())], axis=1)
            out = pd.concat([out, tx_li["DON_AGE"].fillna(tx_li["DON_AGE"].mean())], axis=1)
            out = pd.concat([out, 
                processor.don_type_binary(tx_li["DON_TY"])], axis=1)
            out = pd.concat([out, tx_li["DON_WARM_ISCH_TM_MINS"].fillna(tx_li["DON_WARM_ISCH_TM_MINS"].mean())], axis=1)
            out = pd.concat([out, tx_li["DON_WGT_KG"].fillna(tx_li["DON_WGT_KG"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_ACUTE_REJ_EPISODE"], prefix="REC_ACUTE_REJ_EPISODE")],axis=1)
            out = pd.concat([out, tx_li["REC_AGE_AT_TX"].fillna(tx_li["REC_AGE_AT_TX"].mean())], axis=1)
            out = pd.concat([out, tx_li["REC_BMI"].fillna(tx_li["REC_BMI"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_CMV_STAT"], prefix="REC_CMV_STAT")],axis=1)
            out = pd.concat([out, tx_li["REC_COLD_ISCH_TM"].fillna(tx_li["REC_COLD_ISCH_TM"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_DGN"], prefix="REC_DGN")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_DGN2"], prefix="REC_DGN2")],axis=1)
            out = pd.concat([out, tx_li["REC_DISCHRG_SGPT"].fillna(tx_li["REC_DISCHRG_SGPT"].mean())], axis=1)
            out = pd.concat([out, 
                processor.convert_FUNCSTAT(tx_li["REC_FUNCTN_STAT"])], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_EBV_STAT"].fillna(b'U'), prefix="REC_EBV_STAT")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HBV_ANTIBODY"].fillna(b'U'), prefix="REC_HBV_ANTIBODY")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HBV_SURF_ANTIGEN"].fillna(b'U'), prefix="REC_HBV_SURF_ANTIGEN")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HCV_STAT"].fillna(b'U'), prefix="REC_HCV_STAT")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HIV_STAT"].fillna(b'U'), prefix="REC_HIV_STAT")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_IMMUNO_MAINT_MEDS"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_LIFE_SUPPORT"])],axis=1)
            out = pd.concat([out, tx_li["REC_LIFE_SUPPORT_OTHER"]], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_MED_COND"], prefix="REC_MED_COND")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_PORTAL_VEIN"])],axis=1)
            out = pd.concat([out, tx_li["REC_POSTX_LOS"].fillna(tx_li["REC_POSTX_LOS"].mean())], axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_PREV_ABDOM_SURG"])],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_PRIMARY_PAY"], prefix="REC_PRIMARY_PAY")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_TX_PROCEDURE_TY"], prefix="REC_TX_PROCEDURE_TY")],axis=1)
            out = pd.concat([out, tx_li["REC_VENTILATOR"]], axis=1)
            out = pd.concat([out, tx_li["REC_WARM_ISCH_TM"].fillna(tx_li["REC_WARM_ISCH_TM"].mean())], axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_WORK_INCOME"])],axis=1)
        elif feature == "physiological":
            out = pd.concat([out, tx_li["CANHX_MPXCPT_HCC_APPROVE_IND"]], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_ANGINA"], prefix="CAN_ANGINA")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_ANGINA_CAD"], prefix="CAN_ANGINA_CAD")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_BACTERIA_PERIT"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_CEREB_VASC"])],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["CAN_DIAB_TY"].fillna(998.0), prefix="CAN_DIAB_TY")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_DRUG_TREAT_COPD"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_DRUG_TREAT_HYPERTEN"])],axis=1)
            # out = pd.concat([out,
            #     processor.ordinal_meanpadding(tx_li["CAN_EDUCATION"], {996, 998})], axis=1)
            # out = pd.concat([out, 
            #     processor.ethnicity_binary(tx_li["CAN_ETHNICITY_SRTR"])], axis=1)
            # out = pd.concat([out, 
            #     processor.gender_binary(tx_li["CAN_GENDER"])], axis=1)
            out = pd.concat([out, tx_li["CAN_HGT_CM"].fillna(tx_li["CAN_HGT_CM"].mean())], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_ALBUMIN"].fillna(tx_li["CAN_LAST_ALBUMIN"].mean())], axis=1)
            out = pd.concat([out,
                processor.ordinal_meanpadding(tx_li["CAN_LAST_ASCITES"], {4})], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_BILI"].fillna(tx_li["CAN_LAST_BILI"].mean())], axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_LAST_DIAL_PRIOR_WEEK"])],axis=1)
            out = pd.concat([out,
                processor.ordinal_meanpadding(tx_li["CAN_LAST_ENCEPH"], {4})], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_INR"].fillna(tx_li["CAN_LAST_INR"].mean())], axis=1)
            out = pd.concat([out, 
                processor.convert_MELD(tx_li["CAN_LAST_SRTR_LAB_MELD"])], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_INR"].fillna(tx_li["CAN_LAST_INR"].mean())], axis=1)
            out = pd.concat([out, tx_li["CAN_LAST_SERUM_SODIUM"].fillna(tx_li["CAN_LAST_SERUM_SODIUM"].mean())], axis=1)
            out = pd.concat([out,
                pd.get_dummies(tx_li["CAN_LAST_STAT"], prefix="CAN_LAST_STAT")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_MALIG"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PERIPH_VASC"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PORTAL_VEIN"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PREV_ABDOM_SURG"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_PULM_EMBOL"])],axis=1)
            # out = pd.concat([out, 
            #     pd.get_dummies(tx_li["CAN_RACE_SRTR"], prefix="CAN_RACE_SRTR")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["CAN_TIPSS"])],axis=1)
            out = pd.concat([out, tx_li["CAN_WGT_KG"].fillna(tx_li["CAN_WGT_KG"].mean())], axis=1)
            out = pd.concat([out, tx_li["DON_AGE"].fillna(tx_li["DON_AGE"].mean())], axis=1)
            out = pd.concat([out, 
                processor.don_type_binary(tx_li["DON_TY"])], axis=1)
            out = pd.concat([out, tx_li["DON_WARM_ISCH_TM_MINS"].fillna(tx_li["DON_WARM_ISCH_TM_MINS"].mean())], axis=1)
            out = pd.concat([out, tx_li["DON_WGT_KG"].fillna(tx_li["DON_WGT_KG"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_ACUTE_REJ_EPISODE"], prefix="REC_ACUTE_REJ_EPISODE", dummy_na=True)],axis=1)
            out = pd.concat([out, tx_li["REC_AGE_AT_TX"].fillna(tx_li["REC_AGE_AT_TX"].mean())], axis=1)
            out = pd.concat([out, tx_li["REC_BMI"].fillna(tx_li["REC_BMI"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_CMV_STAT"], prefix="REC_CMV_STAT", dummy_na=True)],axis=1)
            out = pd.concat([out, tx_li["REC_COLD_ISCH_TM"].fillna(tx_li["REC_COLD_ISCH_TM"].mean())], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_DGN"], prefix="REC_DGN")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_DGN2"], prefix="REC_DGN2")],axis=1)
            out = pd.concat([out, tx_li["REC_DISCHRG_SGPT"].fillna(tx_li["REC_DISCHRG_SGPT"].mean())], axis=1)
            out = pd.concat([out, 
                processor.convert_FUNCSTAT(tx_li["REC_FUNCTN_STAT"])], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_EBV_STAT"].fillna(b'U'), prefix="REC_EBV_STAT")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HBV_ANTIBODY"].fillna(b'U'), prefix="REC_HBV_ANTIBODY")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HBV_SURF_ANTIGEN"].fillna(b'U'), prefix="REC_HBV_SURF_ANTIGEN")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HCV_STAT"].fillna(b'U'), prefix="REC_HCV_STAT")],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_HIV_STAT"].fillna(b'U'), prefix="REC_HIV_STAT")],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_IMMUNO_MAINT_MEDS"])],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_LIFE_SUPPORT"])],axis=1)
            out = pd.concat([out, tx_li["REC_LIFE_SUPPORT_OTHER"]], axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_MED_COND"], prefix="REC_MED_COND", dummy_na=True)],axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_PORTAL_VEIN"])],axis=1)
            out = pd.concat([out, tx_li["REC_POSTX_LOS"].fillna(tx_li["REC_POSTX_LOS"].mean())], axis=1)
            out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_PREV_ABDOM_SURG"])],axis=1)
            # out = pd.concat([out, 
            #     pd.get_dummies(tx_li["REC_PRIMARY_PAY"], prefix="REC_PRIMARY_PAY", dummy_na=True)],axis=1)
            out = pd.concat([out, 
                pd.get_dummies(tx_li["REC_TX_PROCEDURE_TY"], prefix="REC_TX_PROCEDURE_TY")],axis=1)
            out = pd.concat([out, tx_li["REC_VENTILATOR"]], axis=1)
            out = pd.concat([out, tx_li["REC_WARM_ISCH_TM"].fillna(tx_li["REC_WARM_ISCH_TM"].mean())], axis=1)
            # out = pd.concat([out, processor.yesnounknown_to_numeric(tx_li["REC_WORK_INCOME"])],axis=1)
        return out

    def extract_infection_XTE(self, tx_lis, txf_li, n_years):
        sets = ["train", "val", "test"]
        Xs, Ts, Es, ids = {}, {}, {}, {}
        
        ###---------------------------------------------------------------
        # for set, tx_li in zip(sets, tx_lis):
        #     num_cohort = len(tx_li.PX_ID.unique())
        #     T, E = np.zeros(num_cohort), np.zeros(num_cohort)
        #     unknowns = []
        #     for i, id in tqdm(enumerate(tx_li.PX_ID), desc='Processing Infection Outcomes', total=len(tx_li)):
        #         row_txf_li = txf_li[txf_li.PX_ID == id]
        #         if row_txf_li.shape[0] == 0:
        #             # no follow-up, censored at transplant time
        #             pass
        #         else:
        #             row_txf_li_infection = row_txf_li[row_txf_li.TFL_ANTIVRL_THERAPY == b'Y']
        #             row_txf_li_no_infection = row_txf_li[row_txf_li.TFL_ANTIVRL_THERAPY == b'N']
        #             if row_txf_li_infection.shape[0] != 0:
        #                 # had infection, find the earliest time
        #                 time = np.nanmin(row_txf_li_infection.time_after_transplant)/365
        #                 if time >= n_years:
        #                     T[i], E[i] = n_years, 0
        #                 elif np.isnan(time):
        #                     T[i], E[i] = np.nan, np.nan
        #                     unknowns.append(id)
        #                 else:
        #                     T[i], E[i] = time, 1
        #             elif row_txf_li_no_infection.shape[0] != 0:
        #                 # no record of infection, find out censored time
        #                 time = np.nanmax(row_txf_li_no_infection.time_after_transplant)/365
        #                 if time >= n_years:
        #                     T[i], E[i] = n_years, 0
        #                 elif np.isnan(time):
        #                     T[i], E[i] = np.nan, np.nan
        #                     unknowns.append(id)
        #                 else:
        #                     T[i], E[i] = time, 0
        #             else:
        #                 # patients with no unknown status for infection
        #                 T[i], E[i] = np.nan, np.nan
        #                 unknowns.append(id)
        ###---------------------------------------------------------------

        for set, tx_li in zip(sets, tx_lis):
            num_cohort = len(tx_li.PX_ID.unique())
            T, E = np.zeros(num_cohort), np.zeros(num_cohort)
            unknowns = []
            for i, id in tqdm(enumerate(tx_li.PX_ID), desc='Processing Infection Outcomes', total=len(tx_li)):
                row_txf_li = txf_li[txf_li.PX_ID == id]
                if row_txf_li.shape[0] == 0:
                    # no follow-up, censored at transplant time
                    pass
                else:
                    row_txf_li_infection = row_txf_li[row_txf_li.TFL_ANTIVRL_THERAPY == b'Y']
                    row_txf_li_no_infection = row_txf_li[row_txf_li.TFL_ANTIVRL_THERAPY == b'N']
                    if row_txf_li_infection.shape[0] != 0:
                        # had infection, find the earliest time
                        time = np.nanmin(row_txf_li_infection.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        # elif time <= 0.7 or np.isnan(time):
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 1
                    elif row_txf_li_no_infection.shape[0] != 0:
                        # no record of infection, find out censored time
                        time = np.nanmax(row_txf_li_no_infection.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 0
                    else:
                        # patients with no unknown status for infection
                        T[i], E[i] = np.nan, np.nan
                        unknowns.append(id)
        
            X = tx_li[~tx_li.PX_ID.isin(unknowns)].copy()
            T = T[~np.isnan(T)]; E = E[~np.isnan(E)]
            ids[set] = np.array(X.PX_ID)
            X = X.drop(["PX_ID"], axis=1)
            Xs[set] = X; Ts[set] = T; Es[set] = E

        cols_to_drop = []
        for i in range(Xs["train"].shape[1]):  # Loop through columns
            if len(Xs["train"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["val"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["test"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        Xs["train"] = Xs["train"].iloc[:, [col for col in range(Xs["train"].shape[1]) if col not in cols_to_drop]]
        Xs["val"] = Xs["val"].iloc[:, [col for col in range(Xs["val"].shape[1]) if col not in cols_to_drop]]
        Xs["test"] = Xs["test"].iloc[:, [col for col in range(Xs["test"].shape[1]) if col not in cols_to_drop]]

        col_names = Xs["train"].columns.values

        mapping = {"train":(np.array(Xs["train"]), Ts["train"], Es["train"]),
                    "val":(np.array(Xs["val"]), Ts["val"], Es["val"]),
                    "test":(np.array(Xs["test"]), Ts["test"], Es["test"])}

        return mapping, col_names, ids


    def extract_cancer_XTE(self, tx_lis, txf_li, n_years):
        sets = ["train", "val", "test"]
        Xs, Ts, Es, ids = {}, {}, {}, {}

        for set, tx_li in zip(sets, tx_lis):
            num_cohort = len(tx_li.PX_ID.unique())
            T, E = np.zeros(num_cohort), np.zeros(num_cohort)
            unknowns = []
            for i, id in tqdm(enumerate(tx_li.PX_ID), desc='Processing Cancer Outcomes', total=len(tx_li)):
                row_txf_li = txf_li[txf_li.PX_ID == id]
                if row_txf_li.shape[0] == 0:
                    # no follow-up, censored at transplant time
                    pass
                else:
                    row_txf_li_cancer = row_txf_li[row_txf_li.TFL_MALIG == b'Y']
                    row_txf_li_no_cancer = row_txf_li[row_txf_li.TFL_MALIG == b'N']
                    if row_txf_li_cancer.shape[0] != 0:
                        # had cancer, find the earliest time
                        time = np.nanmin(row_txf_li_cancer.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 1
                    elif row_txf_li_no_cancer.shape[0] != 0:
                        # no record of cancer, find out censored time
                        time = np.nanmax(row_txf_li_no_cancer.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 0
                    else:
                        # patients with no unknown status for cancer
                        T[i], E[i] = np.nan, np.nan
                        unknowns.append(id)

            X = tx_li[~tx_li.PX_ID.isin(unknowns)].copy()
            T = T[~np.isnan(T)]; E = E[~np.isnan(E)]
            ids[set] = np.array(X.PX_ID)
            X = X.drop(["PX_ID"], axis=1)
            Xs[set] = X; Ts[set] = T; Es[set] = E

        cols_to_drop = []
        for i in range(Xs["train"].shape[1]):  # Loop through columns
            if len(Xs["train"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["val"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["test"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        Xs["train"] = Xs["train"].iloc[:, [col for col in range(Xs["train"].shape[1]) if col not in cols_to_drop]]
        Xs["val"] = Xs["val"].iloc[:, [col for col in range(Xs["val"].shape[1]) if col not in cols_to_drop]]
        Xs["test"] = Xs["test"].iloc[:, [col for col in range(Xs["test"].shape[1]) if col not in cols_to_drop]]

        col_names = Xs["train"].columns.values

        mapping = {"train":(np.array(Xs["train"]), Ts["train"], Es["train"]),
                    "val":(np.array(Xs["val"]), Ts["val"], Es["val"]),
                    "test":(np.array(Xs["test"]), Ts["test"], Es["test"])}

        return mapping, col_names, ids


    def extract_mortality_XTE(self, tx_lis, txf_li, n_years, true_tx_lis, stratify_by = None):
        sets = ["train", "val", "test"]
        Xs, Ts, Es, ids = {}, {}, {}, {}

        for set, tx_li, true_tx_li in zip(sets, tx_lis, true_tx_lis):
            num_cohort = len(tx_li.PX_ID.unique())
            T, E = np.zeros(num_cohort), np.zeros(num_cohort)
            unknowns = []
            for i, id in tqdm(enumerate(tx_li.PX_ID), desc='Processing Mortality Outcomes', total=len(tx_li)):
                row_txf_li = txf_li[txf_li.PX_ID == id]
                row_tx_li = true_tx_li[true_tx_li.PX_ID == id]
                if row_txf_li.shape[0] == 0:
                    # no follow-up, censored at transplant time
                    pass
                else:
                    death_date = list(row_tx_li["PERS_OPTN_DEATH_DT"])[0]

                    if pd.notnull(death_date):
                        # mortality
                        time = list((row_tx_li["PERS_OPTN_DEATH_DT"] - row_tx_li["REC_TX_DT"]).dt.days)[0]/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 1
                    else:
                        # no record of mortality, find out censored time
                        time = np.nanmax(row_txf_li.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 0

            X = tx_li[~tx_li.PX_ID.isin(unknowns)].copy()
            T = T[~np.isnan(T)]; E = E[~np.isnan(E)]
            ids[set] = np.array(X.PX_ID)
            X = X.drop(["PX_ID"], axis=1)
            Xs[set] = X; Ts[set] = T; Es[set] = E

        cols_to_drop = []
        for i in range(Xs["train"].shape[1]):  # Loop through columns
            if len(Xs["train"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["val"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["test"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        Xs["train"] = Xs["train"].iloc[:, [col for col in range(Xs["train"].shape[1]) if col not in cols_to_drop]]
        Xs["val"] = Xs["val"].iloc[:, [col for col in range(Xs["val"].shape[1]) if col not in cols_to_drop]]
        Xs["test"] = Xs["test"].iloc[:, [col for col in range(Xs["test"].shape[1]) if col not in cols_to_drop]]

        col_names = Xs["train"].columns.values

        mapping = {"train":(np.array(Xs["train"]), Ts["train"], Es["train"]),
                    "val":(np.array(Xs["val"]), Ts["val"], Es["val"]),
                    "test":(np.array(Xs["test"]), Ts["test"], Es["test"])}

        return mapping, col_names, ids


    def extract_graft_XTE(self, tx_lis, txf_li, n_years):
        sets = ["train", "val", "test"]
        Xs, Ts, Es, ids = {}, {}, {}, {}

        for set, tx_li in zip(sets, tx_lis):
            num_cohort = len(tx_li.PX_ID.unique())
            T, E = np.zeros(num_cohort), np.zeros(num_cohort)
            unknowns = []
            for i, id in tqdm(enumerate(tx_li.PX_ID), desc='Processing Graft Faliure Outcomes', total=len(tx_li)):
                row_txf_li = txf_li[txf_li.PX_ID == id]
                if row_txf_li.shape[0] == 0:
                    # no follow-up, censored at transplant time
                    pass
                else:
                    row_txf_li["fail_duration"] = (row_txf_li["TFL_FAIL_DT"] - row_txf_li["REC_TX_DT"]).dt.days
                    row_txf_li_with_duration = row_txf_li[pd.notnull(row_txf_li.fail_duration)]

                    if row_txf_li_with_duration.shape[0] != 0:
                        # had failure, and earliest time
                        time = np.nanmin(row_txf_li_with_duration.fail_duration)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 1
                    else:
                        # no record of failure, find out censored time
                        time = np.nanmax(row_txf_li.time_after_transplant)/365
                        if time >= n_years:
                            T[i], E[i] = n_years, 0
                        elif np.isnan(time):
                            T[i], E[i] = np.nan, np.nan
                            unknowns.append(id)
                        else:
                            T[i], E[i] = time, 0

            X = tx_li[~tx_li.PX_ID.isin(unknowns)].copy()
            T = T[~np.isnan(T)]; E = E[~np.isnan(E)]
            ids[set] = np.array(X.PX_ID)
            X = X.drop(["PX_ID"], axis=1)
            Xs[set] = X; Ts[set] = T; Es[set] = E

        cols_to_drop = []
        for i in range(Xs["train"].shape[1]):  # Loop through columns
            if len(Xs["train"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["val"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(Xs["test"].iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        Xs["train"] = Xs["train"].iloc[:, [col for col in range(Xs["train"].shape[1]) if col not in cols_to_drop]]
        Xs["val"] = Xs["val"].iloc[:, [col for col in range(Xs["val"].shape[1]) if col not in cols_to_drop]]
        Xs["test"] = Xs["test"].iloc[:, [col for col in range(Xs["test"].shape[1]) if col not in cols_to_drop]]

        col_names = Xs["train"].columns.values

        mapping = {"train":(np.array(Xs["train"]), Ts["train"], Es["train"]),
                    "val":(np.array(Xs["val"]), Ts["val"], Es["val"]),
                    "test":(np.array(Xs["test"]), Ts["test"], Es["test"])}

        return mapping, col_names, ids

class DatasetValidator():

    def validate_dataframe(self, tx_li, px_ids):
        """
        1. Check the splits for NaN; drop rows/fill accordingly.
        """
        # Check the splits for things that would yield a NaN - specifically,
        # check if there are any columns with only one unique value; these go
        # to infinity when normalizing.
        train_set = tx_li[tx_li['PX_ID'].isin(px_ids["train"])]
        val_set = tx_li[tx_li['PX_ID'].isin(px_ids["val"])]
        test_set = tx_li[tx_li['PX_ID'].isin(px_ids["test"])]

        cols_to_drop = []
        for i in range(tx_li.shape[1]):  # Loop through columns
            if len(train_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(val_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
            if len(test_set.iloc[:, i].unique()) == 1:
                cols_to_drop.append(i)
        tx_li = tx_li.iloc[:, [col for col in range(tx_li.shape[1]) if col not in cols_to_drop]]

        return tx_li

class FeatureProcessor():
    """
    Utility class to gather together feature processing functions.
    """
    def __init__(self):
        MELD_CONVERSION_DIR = "mas/data/reference_tables/meld_conversion.csv"
        # MELD_CONVERSION_DIR = "reference_tables/meld_conversion.csv"
        meld_conversion_cand_liin = pd.read_csv(MELD_CONVERSION_DIR, header=None)
        self.meld_conversion = dict(meld_conversion_cand_liin.values)

        FUNCSTAT_CONVERSION_DIR = "mas/data/reference_tables/funcstat_conversion.csv"
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
        mapping = {b'N': 0, b'Y': 1, b'U': 0.5}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return 0.5 # If we don't know - assume that it's average
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
    print("Running load data!")
    # for outcomes in ["mortality"]:
    #     train, val, test = load_SRTR_static(outcome=outcomes, format="scikit-survival", normalize=["X"])
    #     print(train)
    #     from sksurv.linear_model import CoxnetSurvivalAnalysis
    #     model = CoxnetSurvivalAnalysis()
    #     print("Training...")
    #     model.fit(*train)
    #     print("Evaluating...")
    #     print(model.score(*val))

    # load_SRTR_static_df("cancer", False)
    # load_SRTR_static_df("infection", False)
    load_SRTR_static_df("graft", False)
    # load_SRTR_static_df("mortality", False)
