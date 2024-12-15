import os
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import theano
import lasagne
print(f"Device: {theano.config.device}")

import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_dcox_feature_importance, dynamic_c_index_avg

from mas.models.deepsurv import DeepSurv
from mas.data.load_data import load_SRTR_static_df
from mas.data.load_OPTN import load_OPTN_dataset
from mas.data.load_PD_group import load_PD_dataset
from mas.data.load_data_deepsurv import load_deepsurv
from mas.data.load_dynamic_cox import load_dynamic_cox
from mas.data.load_dcox_delta import load_dcox_delta
from lifelines.utils import concordance_index
from joblib import load, dump


# OPTN 1
# MODEL: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.8096369104223577; tcdi_avg: 0.8499152618794736 

# MODEL: {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.8188183815171283; tcdi_avg: 0.8428322437920007 

# OPTN 2
# MODEL: {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6}
# Validation C-Index: 0.8186068221927926; tcdi_avg: 0.8579722084234752 

# OPTN 3
# MODEL: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7969064605773194; tcdi_avg: 0.8500791277967414 

# OPTN 4
# MODEL: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.7917000406953969; tcdi_avg: 0.8433682548604151 

# OPTN 5
# MODEL: {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6}
# Validation C-Index: 0.7685622197591332; tcdi_avg: 0.7957508944878603 

# OPTN 6 
# MODEL: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.8166512530474731; tcdi_avg: 0.8516052382693113 

# OPTN 7 
# MODEL: {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 0.5, 'n_in': 6}
# Validation C-Index: 0.7963859814156763; tcdi_avg: 0.8481091731113612 

# OPTN 8
# MODEL: {'hidden_layers_sizes': [512, 512, 512], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7233830831704153; tcdi_avg: 0.8162923412468301 

# OPTN 9
# MODEL: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0001, 'dropout': 0.1, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7746651836371299; tcdi_avg: 0.8029059299258615 

# OPTN 10
# MODEL: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.8125761674109983; tcdi_avg: 0.8617176586658251 

# OPTN 11
# MODEL: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}
# Validation C-Index: 0.8113712927852342; tcdi_avg: 0.8428718565494007 


# mortality full
# MODEL 1: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6483228342893026; tcdi_avg: 0.6865209586644788 
# MODEL 2: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.7026422074220507; tcdi_avg: 0.7260427643572094 
# MODEL 3: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.7024210547963166; tcdi_avg: 0.738854900733277 
# MODEL 4: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6959024405654481; tcdi_avg: 0.7301448227254423 
# MODEL 5: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6903536096424723; tcdi_avg: 0.7265733033217875 
# MODEL 6: {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.631777680589892; tcdi_avg: 0.6604246716100751 
# MODEL 7: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6831737981423959; tcdi_avg: 0.7157436150286374 
# MODEL 8: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0001, 'dropout': 0, 'L2_reg': 0.5, 'n_in': 287}
# Validation C-Index: 0.6502082160913328; tcdi_avg: 0.6927951335612309 
# MODEL 9: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6997886251230643; tcdi_avg: 0.7329054723165485 
# MODEL 10: {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6831164817914728; tcdi_avg: 0.7239489729864035 
# MODEL 11: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}
# Validation C-Index: 0.6723732578641656; tcdi_avg: 0.7034161154337988 

# mortality mas
# MODEL 1: {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}
# Validation C-Index: 0.7069752375544252; tcdi_avg: 0.7327121173169465 
# MODEL 2: {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.7112324110255753; tcdi_avg: 0.7336574482777025 
# MODEL 3: {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.0005, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}
# Validation C-Index: 0.6968573636234238; tcdi_avg: 0.738502787523169 
# MODEL 4: {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.0001, 'dropout': 0, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7042559071322564; tcdi_avg: 0.7360109387560029 
# MODEL 5: {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7033054458270303; tcdi_avg: 0.728768821076543 
# MODEL 6: {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.6969942878375827; tcdi_avg: 0.7411678462984492 
# MODEL 7: {'hidden_layers_sizes': [64, 64, 64, 64], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6}
# Validation C-Index: 0.6823483674080759; tcdi_avg: 0.7234206849735595 
# MODEL 8: {'hidden_layers_sizes': [128, 128, 128], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 6}
# Validation C-Index: 0.6675527444545462; tcdi_avg: 0.7174903994094368 
# MODEL 9: {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6}
# Validation C-Index: 0.7118137383756487; tcdi_avg: 0.7416637591629336 
# MODEL 10: {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6}
# Validation C-Index: 0.6809353596281036; tcdi_avg: 0.725120268879744 
# MODEL 11: {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}
# Validation C-Index: 0.6857030717349057; tcdi_avg: 0.7089979232328963 


def train_OPTN_final(OPTN: int, outcome:str, feature:str):
    lasagne.random.set_rng(np.random.RandomState(51))
    out_file = f"experiments/models/deepmas/{outcome}/OPTN/{OPTN}/deepsurv_{feature}_{OPTN}.pkl"

    # graft mas
    # hparams = [ {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6},
    #             {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6},
    #             {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6},
    #             {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 1, 'n_in': 6},
    #             {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6},
    #             {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 0, 'n_in': 6},
    #             {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 0.5, 'n_in': 6},
    #             {'hidden_layers_sizes': [512, 512, 512], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6},
    #             {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0001, 'dropout': 0.1, 'L2_reg': 0, 'n_in': 6},
    #             {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6},
    #             {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}]

    # mortality
    if feature == "mas":
        hparams = [ {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6},
                {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6},
                {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.0005, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6},
                {'hidden_layers_sizes': [16, 16, 16, 16], 'learning_rate': 0.0001, 'dropout': 0, 'L2_reg': 0, 'n_in': 6},
                {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.0005, 'dropout': 0, 'L2_reg': 0, 'n_in': 6},
                {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6},
                {'hidden_layers_sizes': [64, 64, 64, 64], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 1, 'n_in': 6},
                {'hidden_layers_sizes': [128, 128, 128], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 6},
                {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 0, 'n_in': 6},
                {'hidden_layers_sizes': [512, 512], 'learning_rate': 0.001, 'dropout': 0, 'L2_reg': 10, 'n_in': 6},
                {'hidden_layers_sizes': [16, 16, 16], 'learning_rate': 0.001, 'dropout': 0.1, 'L2_reg': 0.5, 'n_in': 6}]
    else:
        hparams = [ {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [128, 128], 'learning_rate': 0.0001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.0001, 'dropout': 0, 'L2_reg': 0.5, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [64, 64], 'learning_rate': 0.0005, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287},
                {'hidden_layers_sizes': [16, 16], 'learning_rate': 0.001, 'dropout': 0.3, 'L2_reg': 10, 'n_in': 287}]

    deepsurv_train, deepsurv_val = load_deepsurv(outcome, feature, OPTN=OPTN)
    _, val_sta = load_OPTN_dataset(OPTN, outcome, "static")
    _, val = load_OPTN_dataset(OPTN, outcome, "dynamic")

    print("training started")

    model = DeepSurv(**hparams[OPTN-1])
    model.train(deepsurv_train, valid_data=deepsurv_val, n_epochs=1000)

    dump(model, out_file)

    print("evaluation started")

    cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
    deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
    deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
    avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")

    print(cindex)
    print(f"tcdi avg: {avg}")


def eval_OPTN(OPTN: int, outcome:str, feature:str):
    out_file = f"experiments/results/{outcome}/OPTN_eval/deepsurv_region-{OPTN}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{OPTN}\n")

    lasagne.random.set_rng(np.random.RandomState(51))
    model_file = f"experiments/models/deepmas/{outcome}/OPTN/{OPTN}/deepsurv_{feature}_{OPTN}.pkl"

    if OPTN <= 11:
        model = load(model_file)
    else:
        model = load(f"experiments/models/deepmas/{outcome}/deepsurv_{feature}.pkl")

    for i in range(11):
        OPTN = i+1
        _, deepsurv_val = load_deepsurv(outcome, feature, OPTN=OPTN)
        _, val_sta = load_OPTN_dataset(OPTN, outcome, "static")
        _, val = load_OPTN_dataset(OPTN, outcome, "dynamic")

        cindex = model.get_concordance_index(deepsurv_val['x'], deepsurv_val['t'], deepsurv_val['e'])
        deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
        deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
        deep_graft_df["RISK"] = model.predict_risk(deepsurv_val['x']).squeeze()
        avg = dynamic_c_index_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")
        print(f"OPTN {OPTN}:")
        print(f"cindex: {cindex}")
        print(f"tcdi avg: {avg}")
        with open(out_file, "a") as f:
            f.write(f"OPTN {OPTN}: C-Index: {cindex}; tcdi_avg: {avg:.4f} \n")


if __name__ == "__main__":
    import fire

    fire.Fire()