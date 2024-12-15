from mas.data.load_data_deepsurv import load_sksurv
from sksurv.ensemble import RandomSurvivalForest
from joblib import dump

# Best full models n_estimators=500, min_samples_split=13, max_depth=9
# Best MAS models n_estimators=500, min_samples_split=20, max_depth=9

# Best full models n_estimators=500, min_samples_split=10, max_depth=5
# Best MAS models n_estimators=500, min_samples_split=20, max_depth=5

def train(model: str = "mas", min_samples_split = 20, max_depth = 5):
    model_file = f"experiments/models/masforest/forest_{model}_regularized.pkl"

    train, val, _ = load_sksurv("graft", model)

    model = RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        max_features="sqrt",
        verbose=100,
        random_state=42,
        n_jobs=50
    )
    print("Training started...")
    model.fit(*train)
    model.set_params(n_jobs=1)
    dump(model, model_file)
    print(model.score(*val))
    print(f"{model} Model Done")


if __name__ == "__main__":
    import fire

    fire.Fire()
