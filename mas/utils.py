import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm
from joblib import Parallel, delayed

from mas.data.load_cr import load_OPTN_crgf
from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_data import load_SRTR_static, load_SRTR_static_df
from mas.data.load_PD_group import load_PD_dataset
from mas.data.load_OPTN import load_OPTN_dataset

DATA_DIR = ""
SPLIT_DIR = ""

def load_dataset(dataset_name: str, tables) -> Dict[str, pd.DataFrame]:
    """
    Loads the dataset into memory.

    Parameters:
    - dataset_name (str) : one of "train", "val", or "test".
    - tables (list) : list of table names.

    Returns:
    - dictionary (str -> pd.DataFrame) : dictionary mapping table names to 
        DataFrames representing those tables.
    """

    with open(f"{SPLIT_DIR}/{dataset_name}_split.txt") as f:
        px_ids = [float(id) for id in f]
    tables_map = {}
    for table in tables:
        # Load the table
        df = pd.read_sas(DATA_DIR + f"{table}.sas7bdat")
        # Filter the table by patient IDs
        df = df[df['PX_ID'].isin(px_ids)]

        tables_map[table] = df.sample(frac=1, random_state=2541)
    return tables_map


def dataset_split(df):
    """
    Split dataset into train, test, val according to the provided PX_IDs.
    Inputs:
    - df (pd.DataFrame) : DataFrame to subset to cohort and split.
    """
    train_PX_IDs = np.loadtxt(SPLIT_DIR + 'train_split.txt', delimiter='\n')
    test_PX_IDs = np.loadtxt(SPLIT_DIR + 'test_split.txt', delimiter='\n')
    val_PX_IDs = np.loadtxt(SPLIT_DIR + 'val_split.txt', delimiter='\n')
    ttv_PX_IDs = np.append(train_PX_IDs, test_PX_IDs)
    ttv_PX_IDs = np.append(ttv_PX_IDs, val_PX_IDs)

    train_df = df[df.PX_ID.isin(train_PX_IDs)]
    test_df = df[df.PX_ID.isin(test_PX_IDs)]
    val_df = df[df.PX_ID.isin(val_PX_IDs)]
    ttv_df = df[df.PX_ID.isin(ttv_PX_IDs)]

    return train_df, test_df, val_df, ttv_df


def load_optn_eval(OPTN: int, mode: str, outcome: str="graft", 
                        deephit: bool=False, normalized: bool=True):
    # load val and test portion of each OPTN region

    # for models trained on all data, when evaluating on OPTN regions,
    # we need to make sure that we are not evaluating its training set,
    # obtain val/test samples from that region.
    _, val, test = load_dynamic_cox(outcome, "ff")
    val_ids, test_ids = val.PX_ID.unique(), test.PX_ID.unique()

    if deephit:
        optn_train, optn_val = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
        optn_train_sta, optn_val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
    else:
        optn_train, optn_val = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="dynamic", 
                                                        normalized=normalized)
        optn_train_sta, optn_val_sta = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="static")

    optn_data, optn_data_sta = pd.concat([optn_train, optn_val]), pd.concat([optn_train_sta, optn_val_sta])

    graft_val, graft_test = optn_data[optn_data.PX_ID.isin(val_ids)], optn_data[optn_data.PX_ID.isin(test_ids)]
    graft_val_sta = optn_data_sta[optn_data_sta.PX_ID.isin(val_ids)]
    graft_test_sta = optn_data_sta[optn_data_sta.PX_ID.isin(test_ids)]

    if mode == "dynamic":
        return graft_val, graft_test
    else:
        return graft_val_sta, graft_test_sta
    

def load_OPTN_train(OPTN: int, mode: str, outcome: str="graft", 
                        deephit: bool=False, normalized: bool=True):
    # load train/test/val portion of each OPTN region

    # for models trained on all data, when evaluating on OPTN regions,
    # we need to make sure that we are not evaluating its training set,
    # obtain val/test samples from that region.
    train, val, test = load_dynamic_cox(outcome, "ff")
    train_ids, val_ids, test_ids = train.PX_ID.unique(), val.PX_ID.unique(), test.PX_ID.unique()

    if deephit:
        optn_train, optn_val = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
        optn_train_sta, optn_val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
    else:
        optn_train, optn_val = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="dynamic", 
                                                        normalized=normalized)
        optn_train_sta, optn_val_sta = load_OPTN_dataset(OPTN=OPTN, outcome=outcome, mode="static")

    optn_data, optn_data_sta = pd.concat([optn_train, optn_val]), pd.concat([optn_train_sta, optn_val_sta])

    graft_train = optn_data[optn_data.PX_ID.isin(train_ids)]
    graft_val, graft_test = optn_data[optn_data.PX_ID.isin(val_ids)], optn_data[optn_data.PX_ID.isin(test_ids)]
    graft_train_sta = optn_data_sta[optn_data_sta.PX_ID.isin(train_ids)]
    graft_val_sta = optn_data_sta[optn_data_sta.PX_ID.isin(val_ids)]
    graft_test_sta = optn_data_sta[optn_data_sta.PX_ID.isin(test_ids)]

    if mode == "dynamic":
        return graft_train, graft_val, graft_test
    else:
        return graft_train_sta, graft_val_sta, graft_test_sta


def calculate_MELD(row):
    """
    """
    bilirubin = row["TFL_TOT_BILI"]
    INR = row["TFL_INR"]
    creatinine = row["TFL_CREAT"]
    meld_calculated = np.round(
        3.78 * np.log(np.clip(bilirubin, 1, None)) + 
        11.2 * np.log(np.clip(INR, 1, None)) +
        9.57 * np.log(np.clip(creatinine, 1, 4)) + 
        6.43
    )
    return list(np.round(np.clip(meld_calculated, 6, 40)))


def calculate_albi(row):
    """
    """
    bilirubin = row["TFL_TOT_BILI"] * 17.1
    albumin = row["TFL_ALBUMIN"] * 10
    albi_calculated = 0.66 * np.log10(bilirubin) - 0.085 * albumin
    return list(albi_calculated)


def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result


def _bootstrap(df_t, risk, t, delta_t_times, delta_t, time_col, i, event_col):
    df_resample = df_t.sample(df_t.shape[0], replace=True)
    risks = np.array(df_resample[risk]).reshape(-1, 1)
    risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )  
    cindex = c_index(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)
    return cindex


def dynamic_c_index(model, df, risk, event_col, time_col, duration_col, t_times, delta_t_times, alpha, iterations, 
                deephit: bool = False, ci: bool = False, n_jobs: int = 10):
    data = pd.DataFrame(index=t_times, columns=delta_t_times)
    upper = pd.DataFrame(index=t_times, columns=delta_t_times)
    lower = pd.DataFrame(index=t_times, columns=delta_t_times)
    text = pd.DataFrame(index=t_times, columns=delta_t_times)
    #  remove NaN values in the risks.
    if deephit:
        for t in delta_t_times:
            df = df[~df[risk+str(t)].isnull()]
    else:
        df = df[~df[risk].isnull()]
    for t in t_times:
        # Make sure the DataFrame contains only:
        # 1. Patients who have not died / been censored prior to the reference time.
        df_t = df[df[time_col] > t]

        if model == "dynamic":
            # Now replace with patients' most recent risks
            # Step 1: Look only at rows with time less than t (we can't look into the future).
            df_t = df_t[df_t[duration_col] <= t]
            # Step 2: take the maximum time (delta_T) per PX_ID as the rows.
            # idx = df_t.groupby("PX_ID")[duration_col].transform(max) == df_t[duration_col]
            # df_t = df_t[idx]
            df_t = df_t.loc[df_t.groupby("PX_ID")[duration_col].idxmax()]

        for i, delta_t in enumerate(delta_t_times):

            df_resample = df_t
            if deephit:
                risks = np.array(df_resample[risk+str(delta_t)]).reshape(-1, 1)
            else:
                risks = np.array(df_resample[risk]).reshape(-1, 1)
            risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
            cindex = c_index(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)

            if ci:
                dynamic_cindices = []
                # for _ in tqdm(range(iterations)):
                #     df_resample = df_t.sample(df_t.shape[0], replace=True)
                #     risks = np.array(df_resample[risk]).reshape(-1, 1)
                #     risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
                
                #     dynamic_cindices.append(
                #         c_index(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)
                #     )
                dynamic_cindices = Parallel(n_jobs=n_jobs)(delayed(_bootstrap)(df_t, risk, t,
                                delta_t_times, delta_t, time_col, i, event_col) for _ in tqdm(range(iterations)))
                lower_ci, median, upper_ci = \
                        np.percentile(dynamic_cindices, [(1-alpha)*100, 50, alpha*100])
                text.loc[t, delta_t] = f"{np.round(cindex, 3)} {np.round(lower_ci, 3), np.round(upper_ci, 3)}"
                data.loc[t, delta_t] = np.round(cindex, 3)
                upper.loc[t, delta_t] = np.round(upper_ci, 3)
                lower.loc[t, delta_t] = np.round(lower_ci, 3)
            else:

                print(f"Time {t}, Delta t {delta_t}, Score {risk}: {np.round(cindex, 4)}")
                data.loc[t, delta_t] = cindex

    if ci:
        data = {"text":text, "data":data, "lower":lower, "upper":upper}
    
    return data
    

def _bootstrap_avg(df, risk, event_col, time_col, duration_col, deephit):
    t_times=[0.5, 1, 3, 5]; delta_t_times=[1, 3, 5, 7]
    data = pd.DataFrame(index=t_times, columns=delta_t_times)
    for t in t_times:

        df_t = df[df[time_col] > t]
        df_t = df_t[df_t[duration_col] <= t]
        df_t = df_t.loc[df_t.groupby("PX_ID")[duration_col].idxmax()]

        for i, delta_t in enumerate(delta_t_times):

            df_resample = df_t.sample(df_t.shape[0], replace=True)
            if deephit:
                risks = np.array(df_resample[risk+str(delta_t)]).reshape(-1, 1)
            else:
                risks = np.array(df_resample[risk]).reshape(-1, 1)
            risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
            cindex = c_index(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)

            data.loc[t, delta_t] = cindex
    avg = data.values.mean()

    return avg




def dynamic_c_index_mean(df, risk, event_col, time_col, duration_col, alpha, iterations, 
                ci: bool = False, n_jobs: int = 10, deephit: bool = False):
    #  remove NaN values in the risks.
    delta_t_times=[1, 3, 5, 7]
    if deephit:
        for t in delta_t_times:
            df = df[~df[risk+str(t)].isnull()]
    else:
        df = df[~df[risk].isnull()]

    avgs = []

    avgs = Parallel(n_jobs=n_jobs)(delayed(_bootstrap_avg)(df, risk, 
                event_col, time_col, duration_col, deephit) for _ in tqdm(range(iterations)))

    lower_ci, median, upper_ci = \
                        np.percentile(avgs, [(1-alpha)*100, 50, alpha*100])
    
    return (lower_ci, upper_ci)


def dynamic_c_index_avg(model, df, risk, event_col, time_col, duration_col, deephit=False,
            t_times=[0.5, 1, 3, 5], delta_t_times=[1, 3, 5, 7], alpha=0.95, iterations=1):
    data = pd.DataFrame(index=t_times, columns=delta_t_times)
    #  remove NaN values in the risks.
    if deephit:
        for t in delta_t_times:
            df = df[~df[risk+str(t)].isnull()]
    else:
        df = df[~df[risk].isnull()]

    for t in t_times:
        # Make sure the DataFrame contains only:
        # 1. Patients who have not died / been censored prior to the reference time.
        df_t = df[df[time_col] > t]

        if model == "dynamic":
            # Now replace with patients' most recent risks
            # Step 1: Look only at rows with time less than t (we can't look into the future).
            df_t = df_t[df_t[duration_col] <= t]
            # Step 2: take the maximum time (delta_T) per PX_ID as the rows.
            # idx = df_t.groupby("PX_ID")[duration_col].transform(max) == df_t[duration_col]
            # df_t = df_t[idx]
            df_t = df_t.loc[df_t.groupby("PX_ID")[duration_col].idxmax()]

        for i, delta_t in enumerate(delta_t_times):

            for _ in range(iterations):
                # df_resample = df_t.sample(df_t.shape[0], replace=True)
                df_resample = df_t
                if deephit:
                    risks = np.array(df_resample[risk+str(delta_t)]).reshape(-1, 1)
                else:
                    risks = np.array(df_resample[risk]).reshape(-1, 1)
                risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
                
                dynamic_cindices = []
                dynamic_cindices.append(
                    c_index(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)
                    )

            lower_ci, cindex, upper_ci = \
                        np.percentile(dynamic_cindices, [(1-alpha)*100, 50, alpha*100])
            data.loc[t, delta_t] = cindex
    
    return data.values.mean()


def c_index_num(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result, Num, Den


def dynamic_c_num_pairs(model, df, risk, event_col, time_col, duration_col, t_times, delta_t_times):
    '''
        get the number of concordant and total pairs
    '''

    data = pd.DataFrame(index=t_times, columns=delta_t_times)
    #  remove NaN values in the risks.
    df = df[~df[risk].isnull()]
    for t in t_times:
        # Make sure the DataFrame contains only:
        # 1. Patients who have not died / been censored prior to the reference time.
        df_t = df[df[time_col] > t]

        if model == "dynamic":
            # Now replace with patients' most recent risks
            # Step 1: Look only at rows with time less than t (we can't look into the future).
            df_t = df_t[df_t[duration_col] <= t]
            # Step 2: take the maximum time (delta_T) per PX_ID as the rows.
            # idx = df_t.groupby("PX_ID")[duration_col].transform(max) == df_t[duration_col]
            # df_t = df_t[idx]
            df_t = df_t.loc[df_t.groupby("PX_ID")[duration_col].idxmax()]

        for i, delta_t in enumerate(delta_t_times):

            df_resample = df_t
            risks = np.array(df_resample[risk]).reshape(-1, 1)
            risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
                
            tu = c_index_num(risks[:, i], np.asarray(df_resample[time_col]), np.asarray(df_resample[event_col]), t+delta_t)
            cindex = tu[0]
            print(f"Time {t}, Delta t {delta_t}, Score {risk}: {np.round(cindex, 4)}")
            data.loc[t, delta_t] = str(int(tu[1])) + "/" + str(int(tu[2]))
    
    return data


def get_dcox_feature_importance(summary, colors, model_name, outcome, feature = "all", save = False):
    coefs = pd.DataFrame(
        np.array(summary["coef"]),
        index = np.array(summary.index),
        columns=["coefficient"]
    )
    non_zero = np.sum(coefs.iloc[:, 0] != 0)
    print("Number of non-zero coefficients: {}".format(non_zero))
    non_zero_coefs = coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(6, 8))
    non_zero_coefs.loc[coef_order][-10:]["coefficient"].plot.barh(ax=ax, legend=False, color=colors[::-1])
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Feature Importances for {outcome}, {model_name}")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.grid(True)
    # plt.tight_layout()
    if save:
        plt.savefig(f"experiments/plots/dynamic/{model_name}-{feature}-{outcome}.png", dpi=500, bbox_inches='tight')
    plt.show()
