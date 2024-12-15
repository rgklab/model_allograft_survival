from mas.data.load_data_deepsurv import load_sksurv
from sksurv.ensemble import RandomSurvivalForest
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_dynamic_cox import load_dynamic_cox
from utils import dynamic_c_index
from joblib import dump, load

def evaluate(model: str = "mas"):
    alpha = 0.95
    NUM_ITERATIONS = 1 # TODO - change to 1000

    t_times = [0, 1, 3, 5, 7, 9]                     # Reference times
    delta_t_times = [1, 3, 5, 7, 9, float("inf")]    # Prediction horizon times

    tt = list(map(lambda x : "t="+str(x), t_times))
    dt = list(map(lambda x : "\u0394t="+str(x), delta_t_times))

    model_file = f"experiments/models/masforest/forest_{model}.pkl"
    out_file = f"dynamic_c_index_eval/c_index_matrix/masforest_{model}.pkl"

    forest = load(model_file)
    forest.set_params(n_jobs=1)

    _, val, _ = load_sksurv("graft", model)
    _, graft_val_sta, _ = load_SRTR_static_df("graft")
    _, graft_val_dyn, _ = load_dynamic_cox("graft", "ff")

    graft_df = graft_val_dyn.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    graft_df = graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    graft_df["RISK"] = forest.predict(val[0])

    masforest_graft_matrix = dynamic_c_index("dynamic", graft_df, "RISK", "EVENT", "TIME", 
                                    "start", t_times, delta_t_times, alpha, NUM_ITERATIONS)
    masforest_graft_matrix.index = tt; masforest_graft_matrix.columns = dt

    print(masforest_graft_matrix.astype(float).round(3))
    dump(masforest_graft_matrix.astype(float).round(3), out_file)


if __name__ == "__main__":
    import fire

    fire.Fire()

