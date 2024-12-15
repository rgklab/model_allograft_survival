from mas.data.load_data_deepsurv import load_sksurv
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_dynamic_cox import load_dynamic_cox
from sksurv.ensemble import RandomSurvivalForest
from utils import get_dcox_feature_importance, dynamic_c_index_avg
from joblib import dump

def train(feature: str):

    out_file = f"experiments/results/forest_{feature}.txt"

    train, val, _ = load_sksurv("graft", feature)
    _, graft_val_sta, _ = load_SRTR_static_df("graft")
    _, dcox_val, _ = load_dynamic_cox("graft", "ff")

    # with open(out_file, "w+") as f:
    with open(out_file, "a") as f:
        f.write(f"Features: {feature}")

    for n_estimator in ([500]):
        for min_samples_split in ([10, 20]):
            for max_depth in ([3, 5]):
                with open(out_file, "a") as f:
                    f.write(f"MODEL: n_estimators={n_estimator}, min_samples_split={min_samples_split}, max_depth={max_depth}\n")
                model = RandomSurvivalForest(
                    n_estimators=n_estimator,
                    min_samples_split=min_samples_split,
                    max_depth=max_depth,
                    max_features="sqrt",
                    verbose=50,
                    random_state=42,
                    n_jobs=40
                )
                print("Training started...")

                try:
                    model.fit(*train)
                except:
                    continue

                model.set_params(n_jobs=1)
                print("Evaluating...")
                try:
                    with open(out_file, "a") as f:

                        deep_graft_df = dcox_val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
                        deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
                        deep_graft_df["RISK"] = model.predict(val[0])
                        avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
                        cindex = model.score(*val)
                        f.write(f"Validation C-Index: {cindex}; tcdi_avg: {avg} \n")
                        print(cindex)
                        print(f"tcdi avg: {avg}")
                except ValueError:
                    with open(out_file, "a") as f:
                        f.write("NaN\n")

    print("Done")


if __name__ == "__main__":
    import fire

    fire.Fire()