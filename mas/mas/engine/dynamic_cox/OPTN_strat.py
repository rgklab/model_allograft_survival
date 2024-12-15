import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_dcox_feature_importance, dynamic_c_index_avg, load_OPTN_train

from mas.data.load_OPTN import load_OPTN_dataset
from lifelines import CoxTimeVaryingFitter
from joblib import load, dump


# graft mas
# OPTN 1
# MODEL: penalizer=0.2, l1_ratio=0.1
# Validation C-Index: 0.9257283157845035; tcdi_avg: 0.8148039084987987 

# OPTN 2
# MODEL: penalizer=0.5, l1_ratio=0.1
# Validation C-Index: 0.9219668199616405; tcdi_avg: 0.8411798601897402 

# OPTN 3
# MODEL: penalizer=0.5, l1_ratio=0.1
# Validation C-Index: 0.9145458466573352; tcdi_avg: 0.8331807939819973 

# OPTN 4
# MODEL: penalizer=0.1, l1_ratio=0.5
# Validation C-Index: 0.926784093476123; tcdi_avg: 0.852795024126568 

# OPTN 5
# MODEL: penalizer=0.5, l1_ratio=0
# Validation C-Index: 0.897439432256906; tcdi_avg: 0.7741860781826397 

# OPTN 6 
# MODEL: penalizer=0.5, l1_ratio=0.1
# Validation C-Index: 0.9043051377268027; tcdi_avg: 0.8396252694065383 

# OPTN 7 
# MODEL: penalizer=0.1, l1_ratio=0.5
# Validation C-Index: 0.9274486068798786; tcdi_avg: 0.8124223939214072 

# OPTN 8
# MODEL: penalizer=0.5, l1_ratio=0.1
# Validation C-Index: 0.9196452047704584; tcdi_avg: 0.8572096096417507 

# OPTN 9
# MODEL: penalizer=0.1, l1_ratio=0.1
# Validation C-Index: 0.8976134728525087; tcdi_avg: 0.8460048841093825 

# OPTN 10
# MODEL: penalizer=0.1, l1_ratio=0.5
# Validation C-Index: 0.9209419598387424; tcdi_avg: 0.8209437486188613 

# OPTN 11
# MODEL: penalizer=0.01, l1_ratio=0.7
# Validation C-Index: 0.9363313694755239; tcdi_avg: 0.8484666624918236 


# graft full
# MODEL 1: penalizer=0.2, l1_ratio=0.1
# Validation C-Index: 0.9262772085556448; tcdi_avg: 0.8177953285187936 
# MODEL 2: penalizer=0.1, l1_ratio=0.1
# Validation C-Index: 0.9164915161802283; tcdi_avg: 0.8410852278717251 
# MODEL 3: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.9227996939227133; tcdi_avg: 0.8511030681582022 
# MODEL 4: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.9256001536653099; tcdi_avg: 0.8646030618508187 
# MODEL 5: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.9084252386795558; tcdi_avg: 0.7949673469672451 
# MODEL 6: penalizer=0.01, l1_ratio=0.3
# Validation C-Index: 0.9029272821284197; tcdi_avg: 0.8178956910493275 
# MODEL 7: penalizer=0.1, l1_ratio=0.5
# Validation C-Index: 0.9473697983274286; tcdi_avg: 0.8514665878686847 
# MODEL 8: penalizer=0.1, l1_ratio=0.5
# Validation C-Index: 0.9173832779326169; tcdi_avg: 0.8527184365286987 
# MODEL 9: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.9054431209610816; tcdi_avg: 0.8503950832404393 
# MODEL 10: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.9263621187703568; tcdi_avg: 0.8367877895535332
# MODEL 11: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.9381979352467434; tcdi_avg: 0.8558746141308344 

# mortality full
# MODEL 1: penalizer=0.5, l1_ratio=0
# Validation C-Index: 0.8600995644141809; tcdi_avg: 0.7491813669151126 
# MODEL 2: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.8654063121755334; tcdi_avg: 0.7558154775568527 
# MODEL 3: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.8646478406970518; tcdi_avg: 0.7715570753142882 
# MODEL 4: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.8722096691880824; tcdi_avg: 0.7670377970070926 
# MODEL 5: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.8710492321510499; tcdi_avg: 0.7598539507609726 
# MODEL 6: penalizer=0.1, l1_ratio=0.1
# Validation C-Index: 0.8600466728541311; tcdi_avg: 0.7657514498720609 
# MODEL 7: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.865079283059296; tcdi_avg: 0.7572792729544776 
# MODEL 8: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.8590698952160434; tcdi_avg: 0.7447671558159064 
# MODEL 9: penalizer=0.1, l1_ratio=0
# Validation C-Index: 0.8752502749811878; tcdi_avg: 0.7750208722198969 
# MODEL 10: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.8601727788324525; tcdi_avg: 0.7656512015073211 
# MODEL 11: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.8603671614505521; tcdi_avg: 0.7470141978438505 

# mortality mas
# MODEL 1: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.8109077863937693; tcdi_avg: 0.7222759214940987 
# MODEL 2: penalizer=0.1, l1_ratio=0.1
# Validation C-Index: 0.8218680376420685; tcdi_avg: 0.7292680246799786 
# MODEL 3: penalizer=0.1, l1_ratio=0.7
# Validation C-Index: 0.8196366242339315; tcdi_avg: 0.7345663845013122
# MODEL 4: penalizer=0.2, l1_ratio=0.5
# Validation C-Index: 0.8217645266292255; tcdi_avg: 0.73476542808295 
# MODEL 5: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.807668491733118; tcdi_avg: 0.7215625752887636 
# MODEL 6: penalizer=0.5, l1_ratio=0
# Validation C-Index: 0.8242577266702448; tcdi_avg: 0.7478581423983842 
# MODEL 7: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.808657329716961; tcdi_avg: 0.7042274432909654 
# MODEL 8: penalizer=0.2, l1_ratio=0
# Validation C-Index: 0.7992775008830109; tcdi_avg: 0.7075976149516368 
# MODEL 9: penalizer=0.1, l1_ratio=0.1
# Validation C-Index: 0.8453128226763535; tcdi_avg: 0.7412858974539951 
# MODEL 10: penalizer=1, l1_ratio=0
# Validation C-Index: 0.815668727845517; tcdi_avg: 0.7219531363072673 
# MODEL 11: penalizer=0.5, l1_ratio=0.5
# Validation C-Index: 0.8046582346587201; tcdi_avg: 0.709934281914454 

def train_OPTN_final(OPTN: int, outcome:str, feature:str):

    out_file = f"experiments/models/dynamic_cox/{outcome}/OPTN/{OPTN}/dcox_{feature}_{OPTN}.pkl"

    # mortality
    if feature == "mas":
        # hparams = [ {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0.1},
        #         {"penalizer": 0.1, "l1_ratio": 0.7},
        #         {"penalizer": 0.2, "l1_ratio": 0.5},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.5, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0.1},
        #         {"penalizer": 1, "l1_ratio": 0},
        #         {"penalizer": 0.5, "l1_ratio": 0.5}]
        # graft mas
        hparams = [ {"penalizer": 0.2, "l1_ratio": 0.1},
                {"penalizer": 0.5, "l1_ratio": 0.1},
                {"penalizer": 0.5, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0.5},
                {"penalizer": 0.5, "l1_ratio": 0},
                {"penalizer": 0.5, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0.5},
                {"penalizer": 0.5, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0.5},
                {"penalizer": 0.01, "l1_ratio": 0.7}]
    else:
        # mortality
        # hparams = [ {"penalizer": 0.5, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0.1},
        #         {"penalizer": 0.1, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0},
        #         {"penalizer": 0.1, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0},
        #         {"penalizer": 0.2, "l1_ratio": 0}]
        hparams = [ {"penalizer": 0.2, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0.1},
                {"penalizer": 0.1, "l1_ratio": 0},
                {"penalizer": 0.2, "l1_ratio": 0},
                {"penalizer": 0.2, "l1_ratio": 0},
                {"penalizer": 0.01, "l1_ratio": 0.3},
                {"penalizer": 0.1, "l1_ratio": 0.5},
                {"penalizer": 0.1, "l1_ratio": 0.5},
                {"penalizer": 0.2, "l1_ratio": 0},
                {"penalizer": 0.2, "l1_ratio": 0},
                {"penalizer": 0.2, "l1_ratio": 0}]

    # _, graft_val_sta = load_OPTN_dataset(OPTN, outcome, "static")
    # train, val = load_OPTN_dataset(OPTN, outcome, "dynamic")

    train, val, test = load_OPTN_train(OPTN, "dynamic", outcome)
    train_sta, val_sta, test_sta = load_OPTN_train(OPTN, "static", outcome)
    # use both val and test as one set due to small sample size
    val = pd.concat([val, test])
    graft_val_sta = pd.concat([val_sta, test_sta])

    if feature == "full":
        pass
    elif feature == "mas":
        variables = ["TFL_CREAT", "TFL_INR", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALBUMIN"]
        train = train[["PX_ID", "EVENT", "start", "stop"] + variables]
        val = val[["PX_ID", "EVENT", "start", "stop"] + variables]

    print("training started")

    model = CoxTimeVaryingFitter(**hparams[OPTN-1])
    model.fit(train, id_col="PX_ID", event_col="EVENT", start_col="start", stop_col="stop", show_progress=True)

    dump(model, out_file)

    print("evaluation started")

    # eval tdci avg
    dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
    dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    dcox_graft_df["RISK"] = model.predict_partial_hazard(val)

    avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
    print(f"tdci avg: {avg}")


def eval_OPTN(OPTN: int, outcome:str, feature:str):
    out_file = f"experiments/results/{outcome}/OPTN_eval/dcox_region-{OPTN}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{OPTN}\n")
    model_file = f"experiments/models/dynamic_cox/{outcome}/OPTN/{OPTN}/dcox_{feature}_{OPTN}.pkl"

    if OPTN <= 11:
        model = load(model_file)
    else:
        model = load(f"experiments/models/mas/{outcome}/cox_{feature}.pkl")

    for i in range(11):
        OPTN = i+1
        # _, graft_val_sta = load_OPTN_dataset(OPTN, outcome, "static")
        # _, val = load_OPTN_dataset(OPTN, outcome, "dynamic")
        train, val, test = load_OPTN_train(OPTN, "dynamic", outcome)
        train_sta, val_sta, test_sta = load_OPTN_train(OPTN, "static", outcome)
        # use both val and test as one set due to small sample size
        val = pd.concat([val, test])
        graft_val_sta = pd.concat([val_sta, test_sta])

        dcox_graft_df = val.drop(columns=["EVENT"]).merge(graft_val_sta, on="PX_ID", how="left")
        dcox_graft_df = dcox_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
        dcox_graft_df["RISK"] = model.predict_partial_hazard(val)

        avg = dynamic_c_index_avg("dynamic", dcox_graft_df, "RISK", "EVENT", "TIME", "start")
        print(f"OPTN {OPTN}:")
        print(f"tcdi avg: {avg}")
        with open(out_file, "a") as f:
            f.write(f"OPTN {OPTN}: tcdi_avg: {avg:.4f} \n")
            # f.write(f"{avg:.3f} \n")


if __name__ == "__main__":
    import fire

    fire.Fire()