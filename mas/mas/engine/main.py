import pandas as pd
from pycox.models import DeepHit

from mas.models.deephit.model import CauseSpecificNet
from mas.data.load_cr import load_deephit, load_dcox_gfcr, load_SRTR_static_gfcr
from mas.data.load_OPTN import load_OPTN_dataset
from mas.data.load_cr import load_OPTN_crgf, load_optn_eval_deephit
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_data_deepsurv import load_deepsurv, load_sksurv, load_optn_eval_deepsurv, load_optn_eval_sksurv
from utils import dynamic_c_index, dynamic_c_index_mean, dynamic_c_index_avg, load_optn_eval
from utils import calculate_MELD, calculate_albi
from joblib import dump, load


def load_evaluate_dataset(model: str, outcome: str, feature: str, dataset: str, OPTN = None):
    """
    model: [cox, deepsurv, deephit, forest]
    feature: [mas, full]
    dataset: [val, test]

    for OPTN datasets, load both val and test to increase sample size for each region
    """
    deephit = True if model == "deephit" else False
    if OPTN:
        if model == "albi" or model == "meld":
            graft_val, graft_test = load_optn_eval(OPTN, "dynamic", outcome, deephit, False)
        else:
            graft_val, graft_test = load_optn_eval(OPTN, "dynamic", outcome, deephit)
        graft_val_sta, graft_test_sta = load_optn_eval(OPTN, "static", outcome, deephit)
    else:
        if model == "deephit":
            _, graft_val, graft_test = load_dcox_gfcr()
            _, graft_val_sta, graft_test_sta = load_SRTR_static_gfcr()
        else:
            if model == "albi" or model == "meld":
                _, graft_val, graft_test = load_dynamic_cox(outcome, "ff", normalized=False)
            else:
                _, graft_val, graft_test = load_dynamic_cox(outcome, "ff", normalized=True)
            _, graft_val_sta, graft_test_sta = load_SRTR_static_df(outcome)

    if feature == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
    elif feature == "meld":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_TOT_BILI"]
    elif feature == "meaf":
        variables = ["TFL_SGPT", "TFL_INR", "TFL_TOT_BILI"]
    elif feature == "albi":
        variables = ["TFL_TOT_BILI", "TFL_ALBUMIN"]
    
    graft_val = graft_val[["PX_ID", "EVENT", "start", "stop"] + variables]
    graft_test = graft_test[["PX_ID", "EVENT", "start", "stop"] + variables]

    val = graft_val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    val = val[["PX_ID", "EVENT", "TIME", "start"]]
    test = graft_test.drop(columns=["EVENT"]).merge(graft_test_sta, on="PX_ID", how="left")
    test = test[["PX_ID", "EVENT", "TIME", "start", "stop"]]

    if model == "cox":
        saved_model = load(f"experiments/models/mas/{outcome}/cox_{feature}.pkl")
        val["RISK"] = saved_model.predict_partial_hazard(graft_val)
        test["RISK"] = saved_model.predict_partial_hazard(graft_test)
    elif model == "albi":
        val["RISK"] = calculate_albi(graft_val)
        test["RISK"] = calculate_albi(graft_test)
    elif model == "meld":
        val["RISK"] = calculate_MELD(graft_val)
        test["RISK"] = calculate_MELD(graft_test)
    elif model == "deepsurv":
        saved_model = load(f"experiments/models/deepmas/{outcome}/deepsurv_{feature}.pkl")
        if OPTN:
            deepsurv_val, deepsurv_test = load_optn_eval_deepsurv(outcome, feature, OPTN)
        else:
            _, deepsurv_val, deepsurv_test = load_deepsurv(outcome, feature)  
        val["RISK"] = saved_model.predict_risk(deepsurv_val['x']).squeeze()
        test["RISK"] = saved_model.predict_risk(deepsurv_test['x']).squeeze()
    elif model == "forest":
        saved_model = load(f"experiments/models/masforest/forest_{feature}.pkl")
        if OPTN:
            forest = load_optn_eval_sksurv(outcome, feature, OPTN)
            val = pd.concat([val, test])
            val = val.reset_index(drop=True)
            val["RISK"] = saved_model.predict(forest[0])
        else:
            _, forest_val, forest_test = load_sksurv("graft", feature)
            eval(dataset)["RISK"] = saved_model.predict(eval(f"forest_{dataset}[0]"))
        # val["RISK"] = saved_model.predict(forest_val[0])
        # test["RISK"] = saved_model.predict(forest_test[0])
    elif model == "deephit":
        saved_model = DeepHit(f"experiments/models/deephit/deephit_{feature}_True.pkl")
        
        if OPTN:
            val_set, test_set = load_optn_eval_deephit(feature, OPTN)
        else:
            _, val_set, test_set = load_deephit(feature)
        x_val, _ = val_set; x_test, _ = test_set

        cif_val = saved_model.predict_cif(x_val)
        cif1 = pd.DataFrame(cif_val[0], saved_model.duration_index).transpose()
        cif1.columns = cif1.columns.astype("int")
        cif1.columns = ["RISK"+str(i) for i in cif1.columns]

        cif_test = saved_model.predict_cif(x_test)
        cif2 = pd.DataFrame(cif_test[0], saved_model.duration_index).transpose()
        cif2.columns = cif2.columns.astype("int")
        cif2.columns = ["RISK"+str(i) for i in cif2.columns]

        for i in [1, 3, 5, 7]:
            val["RISK"+str(i)] = cif1["RISK"+str(i)]
            test["RISK"+str(i)] = cif2["RISK"+str(i)]
        val.loc[val.EVENT==2, "EVENT"] = 0
        test.loc[test.EVENT==2, "EVENT"] = 0
    else:
        print("No such model provided!")

    if OPTN:
        return (val, test)
    else:
        return eval(dataset)


def mean_tdci(model: str, outcome: str, feature: str, dataset: str = "test"):
    """
    model: [cox, deepsurv, deephit, forest]
    outcome: [graft, mortality]
    feature: [mas, full]
    dataset: [val, test]
    """
    out_file = f"experiments/results/model_{dataset}_{feature}_mean_tdci.txt"
    data = load_evaluate_dataset(model, outcome, feature, dataset)

    deephit = True if model == "deephit" else False

    ci = dynamic_c_index_mean(data, "RISK", "EVENT", "TIME", "start", alpha=0.95, iterations=1000, n_jobs=40, deephit=deephit)
    print(f"{ci}")
    avg = dynamic_c_index_avg("dynamic", data, "RISK", "EVENT", "TIME", "start", deephit=deephit)

    print(f"{avg} {ci}")
    with open(out_file, "a") as f:
        f.write(f"{model} ({feature}, {dataset}): {avg}, {ci} \n")


def optn_mean_tdci(model: str, outcome: str, feature: str, OPTN : int, dataset: str = "test"):
    """
    model: [cox, deepsurv, deephit, forest]
    outcome: [graft, mortality]
    feature: [mas, full]
    dataset: [val, test]
    """
    out_file = f"experiments/results/OPTN_model_{dataset}_{feature}_mean_tdci.txt"
    val, test = load_evaluate_dataset(model, outcome, feature, dataset, OPTN=OPTN)

    if model == "forest":
        # val contains both val and test to save time
        data = val
    else:
        data = pd.concat([val, test])

    data = data.reset_index(drop=True)

    deephit = True if model == "deephit" else False

    ci = dynamic_c_index_mean(data, "RISK", "EVENT", "TIME", "start", alpha=0.95, iterations=1000, n_jobs=50, deephit=deephit)
    print(f"{ci}")
    avg = dynamic_c_index_avg("dynamic", data, "RISK", "EVENT", "TIME", "start", deephit=deephit)

    print(f"{avg} {ci}")
    with open(out_file, "a") as f:
        f.write(f"OPTN {OPTN} {model} ({feature}, {dataset}): {avg}, {ci} \n")


def tdci_matrix(model: str, feature: str, dataset: str, ci: bool = False):
    data = load_evaluate_dataset(model, feature, dataset)

    alpha = 0.95; NUM_ITERATIONS = 1000
    t_times = [0.5, 1, 3, 5, 7, 9]                     # Reference times
    delta_t_times = [1, 3, 5, 7, 9, float("inf")]      # Prediction horizon times

    tt = list(map(lambda x : "t="+str(x), t_times))
    dt = list(map(lambda x : "\u0394t="+str(x), delta_t_times))

    if ci:
        matrix = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                        "start", t_times, delta_t_times, alpha, NUM_ITERATIONS, ci=True, n_jobs=20)
    else:
        matrix = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                    "start", t_times, delta_t_times, alpha, NUM_ITERATIONS, ci=False, n_jobs=20)
        matrix = matrix.astype(float).round(3)


    matrix.index = tt; matrix.columns = dt
    matrix = matrix.astype(float).round(3)
    print(matrix) 
    dump(matrix, f"experiments/results/tdci_matrix/{model}_{feature}_{dataset}_{ci}.pkl")


def get_x_year_gf(model, outcome, feature, dataset = "test"):
    # this function is used to get numbers for the line ci plot

    data = load_evaluate_dataset(model = model, outcome = outcome, feature = feature, dataset = dataset)
    alpha = 0.95; NUM_ITERATIONS = 1000

    t_times = [0.5, 1, 3, 5, 7, 9]                     # Reference times
    delta_t_times = [1, 3, 5, 7, 9, 10]      # Prediction horizon times

    data_dict = dynamic_c_index("dynamic", data, "RISK", "EVENT", "TIME", 
                                    "start", t_times, delta_t_times, alpha, NUM_ITERATIONS, ci=True, n_jobs=20)

    dump(data_dict, f"experiments/results/plotting/line_plot/{model}_{outcome}_{feature}_{dataset}.pkl")


if __name__ == "__main__":
    import fire

    fire.Fire()