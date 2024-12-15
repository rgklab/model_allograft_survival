import pandas as pd
import torch
import wandb
import numpy as np
from utils import tdci_crgf_avg

import torchtuples as tt
from functools import partial
from pycox.models import DeepHit
from mas.data.load_cr import load_SRTR_static_gfcr, load_dcox_gfcr, load_deephit, load_OPTN_crgf
from utils import tdci_crgf_avg
from joblib import load, dump

class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out

#  full
# MODEL 1: {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 1,
#           'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.3}
# tcdi_avg: 0.8087
# MODEL 2: {'alpha': 0.8, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 2,
#           'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.1}
# tcdi_avg: 0.8357
# MODEL 3: {'alpha': 0.2, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 3, 'layers_shared': 1,
#           'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 64, 'sigma': 0.1}
# tcdi_avg: 0.8406
# MODEL 4: {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 1,
#           'lr': 0.001, 'nodes_indiv': 256, 'nodes_shared': 256, 'sigma': 0.3}
# tcdi_avg: 0.8369
# MODEL 5: {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 2,
#           'lr': 0.001, 'nodes_indiv': 256, 'nodes_shared': 512, 'sigma': 0.5}
# tcdi_avg: 0.8040
# MODEL 6: {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0, 'layers_indiv': 3, 'layers_shared': 3,
#           'lr': 0.01, 'nodes_indiv': 512, 'nodes_shared': 512, 'sigma': 0.3}
# tcdi_avg: 0.7992
# MODEL 7: {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 1, 'layers_shared': 3,
#           'lr': 0.0005, 'nodes_indiv': 64, 'nodes_shared': 256, 'sigma': 0.1}
# tcdi_avg: 0.8174
# MODEL 8: {'alpha': 0.5, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.1, 'layers_indiv': 3, 'layers_shared': 2,
#           'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 16, 'sigma': 0.1}
# tcdi_avg: 0.7839
# MODEL 9: {'alpha': 0.5, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.5, 'layers_indiv': 3, 'layers_shared': 1,
#           'lr': 0.005, 'nodes_indiv': 64, 'nodes_shared': 64, 'sigma': 0.3}
# tcdi_avg: 0.7799
# MODEL 10: {'alpha': 0.2, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0.1, 'layers_indiv': 1, 'layers_shared': 1,
#           'lr': 0.001, 'nodes_indiv': 128, 'nodes_shared': 16, 'sigma': 0.3}
# tcdi_avg: 0.8355
# MODEL 11: {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.5, 'layers_indiv': 1, 'layers_shared': 1,
#           'lr': 0.001, 'nodes_indiv': 512, 'nodes_shared': 64, 'sigma': 0.1}
# tcdi_avg: 0.8309


#  mas
# MODEL 1: {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 2, 'layers_shared': 2,
#           'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 64, 'sigma': 0.1}
# tcdi_avg: 0.8548
# MODEL 2: {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 1,
#           'lr': 0.001, 'nodes_indiv': 64, 'nodes_shared': 16, 'sigma': 0.3}
# tcdi_avg: 0.8577
# MODEL 3: {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0, 'layers_indiv': 3, 'layers_shared': 1,
#           'lr': 0.0005, 'nodes_indiv': 16, 'nodes_shared': 64, 'sigma': 0.3}
# tcdi_avg: 0.8508
# MODEL 4: {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.1, 'layers_indiv': 1, 'layers_shared': 3,
#           'lr': 0.0005, 'nodes_indiv': 512, 'nodes_shared': 512, 'sigma': 0.3}
# tcdi_avg: 0.8398
# MODEL 5: {'alpha': 0.8, 'batch_norm': 1, 'batch_size': 128, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 2,
#           'lr': 0.01, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.5}
# tcdi_avg: 0.8078
# MODEL 6: {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0, 'layers_indiv': 1, 'layers_shared': 2,
#           'lr': 0.005, 'nodes_indiv': 256, 'nodes_shared': 256, 'sigma': 0.5}
# tcdi_avg: 0.8611
# MODEL 7: {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.1, 'layers_indiv': 2, 'layers_shared': 3,
#           'lr': 0.005, 'nodes_indiv': 64, 'nodes_shared': 16, 'sigma': 0.5}
# tcdi_avg: 0.8481
# MODEL 8: {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 2,
#           'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 256, 'sigma': 0.3}
# tcdi_avg: 0.8241
# MODEL 9: {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0.3, 'layers_indiv': 1, 'layers_shared': 1,
#           'lr': 0.01, 'nodes_indiv': 256, 'nodes_shared': 16, 'sigma': 0.3}
# tcdi_avg: 0.8037
# MODEL 10: {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.1, 'layers_indiv': 3, 'layers_shared': 2,
#           'lr': 0.01, 'nodes_indiv': 256, 'nodes_shared': 64, 'sigma': 0.3}
# tcdi_avg: 0.8692
# MODEL 11: {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 1,
#           'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 16, 'sigma': 0.3}
# tcdi_avg: 0.8397


def train_OPTN_final(OPTN: int, feature:str):
    np.random.seed(1234)
    _ = torch.manual_seed(1234)
    out_file = f"experiments/models/deephit/OPTN/{OPTN}/deepsurv_{feature}_{OPTN}.pkl"

    if feature == "mas":
        hparams = [ {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 2, 'layers_shared': 2,
                    'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 64, 'sigma': 0.1},
                {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 1,
                   'lr': 0.001, 'nodes_indiv': 64, 'nodes_shared': 16, 'sigma': 0.3},
                {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0, 'layers_indiv': 3, 'layers_shared': 1,
                   'lr': 0.0005, 'nodes_indiv': 16, 'nodes_shared': 64, 'sigma': 0.3},
                {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.1, 'layers_indiv': 1, 'layers_shared': 3,
                   'lr': 0.0005, 'nodes_indiv': 512, 'nodes_shared': 512, 'sigma': 0.3},
                {'alpha': 0.8, 'batch_norm': 1, 'batch_size': 128, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 2,
                   'lr': 0.01, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.5},
                {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0, 'layers_indiv': 1, 'layers_shared': 2,
                   'lr': 0.005, 'nodes_indiv': 256, 'nodes_shared': 256, 'sigma': 0.5},
                {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.1, 'layers_indiv': 2, 'layers_shared': 3,
                   'lr': 0.005, 'nodes_indiv': 64, 'nodes_shared': 16, 'sigma': 0.5},
                {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 256, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 2,
                   'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 256, 'sigma': 0.3},
                {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0.3, 'layers_indiv': 1, 'layers_shared': 1,
                   'lr': 0.01, 'nodes_indiv': 256, 'nodes_shared': 16, 'sigma': 0.3},
                {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.1, 'layers_indiv': 3, 'layers_shared': 2,
                   'lr': 0.01, 'nodes_indiv': 256, 'nodes_shared': 64, 'sigma': 0.3},
                {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 1,
                   'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 16, 'sigma': 0.3}]
    else:
        hparams = [ {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 1,
                    'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.3},
                {'alpha': 0.8, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 2, 'layers_shared': 2,
                   'lr': 0.005, 'nodes_indiv': 128, 'nodes_shared': 512, 'sigma': 0.1},
                {'alpha': 0.2, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 3, 'layers_shared': 1,
                   'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 64, 'sigma': 0.1},
                {'alpha': 0.5, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 1,
                   'lr': 0.001, 'nodes_indiv': 256, 'nodes_shared': 256, 'sigma': 0.3},
                {'alpha': 0.8, 'batch_norm': 0, 'batch_size': 32, 'dropout': 0.3, 'layers_indiv': 3, 'layers_shared': 2,
                   'lr': 0.001, 'nodes_indiv': 256, 'nodes_shared': 512, 'sigma': 0.5},
                {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0, 'layers_indiv': 3, 'layers_shared': 3,
                   'lr': 0.01, 'nodes_indiv': 512, 'nodes_shared': 512, 'sigma': 0.3},
                {'alpha': 0.9, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.5, 'layers_indiv': 1, 'layers_shared': 3,
                   'lr': 0.0005, 'nodes_indiv': 64, 'nodes_shared': 256, 'sigma': 0.1},
                {'alpha': 0.5, 'batch_norm': 1, 'batch_size': 64, 'dropout': 0.1, 'layers_indiv': 3, 'layers_shared': 2,
                   'lr': 0.01, 'nodes_indiv': 16, 'nodes_shared': 16, 'sigma': 0.1},
                {'alpha': 0.5, 'batch_norm': 1, 'batch_size': 32, 'dropout': 0.5, 'layers_indiv': 3, 'layers_shared': 1,
                   'lr': 0.005, 'nodes_indiv': 64, 'nodes_shared': 64, 'sigma': 0.3},
                {'alpha': 0.2, 'batch_norm': 0, 'batch_size': 64, 'dropout': 0.1, 'layers_indiv': 1, 'layers_shared': 1,
                   'lr': 0.001, 'nodes_indiv': 128, 'nodes_shared': 16, 'sigma': 0.3},
                {'alpha': 0.9, 'batch_norm': 0, 'batch_size': 128, 'dropout': 0.5, 'layers_indiv': 1, 'layers_shared': 1,
                   'lr': 0.001, 'nodes_indiv': 512, 'nodes_shared': 64, 'sigma': 0.1}]

    train, val  = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
    _, val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
    train, val_set = load_deephit(feature, OPTN=OPTN)
    x_train, y_train = train
    x_val, _ = val_set

    in_features = x_train.shape[1]
    num_risks = y_train[1].max()
    cuts = np.arange(20, dtype="float64")
    out_features = 20

    epochs = 512
    batch_norm = bool(hparams[OPTN-1]["batch_norm"])
    dropout = hparams[OPTN-1]["dropout"]
    num_nodes_shared = hparams[OPTN-1]["nodes_shared"]
    num_layers_shared = hparams[OPTN-1]["layers_shared"]
    num_nodes_indiv = hparams[OPTN-1]["nodes_indiv"]
    num_layers_indiv = hparams[OPTN-1]["layers_indiv"]
    batch_size = hparams[OPTN-1]["batch_size"]
    lr = hparams[OPTN-1]["lr"]
    alpha = hparams[OPTN-1]["alpha"]
    sigma = hparams[OPTN-1]["sigma"]
    layer_shared = [num_nodes_shared for i in range(num_layers_shared)]
    layer_indiv = [num_nodes_indiv for i in range(num_layers_indiv)]

    net = CauseSpecificNet(in_features, layer_shared, layer_indiv, num_risks,
                        out_features, batch_norm, dropout)
    callbacks = [tt.callbacks.EarlyStoppingCycle()]
    verbose = False

    # if use Adam, change callback to EarlyStopping instead of cycle
    # optimizer = tt.optim.Adam(lr=lr)
    optimizer = tt.optim.AdamWR(lr=lr, decoupled_weight_decay=0.01,
                            cycle_eta_multiplier=0.8)
    model = DeepHit(net, optimizer, alpha=alpha, sigma=sigma,
                    duration_index=cuts)
    print("training started")
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val_set)

    print("evaluation started")
    cif = model.predict_cif(x_val)
    cif1 = pd.DataFrame(cif[0], model.duration_index).transpose()
    cif1.columns = cif1.columns.astype("int")
    cif1.columns = ["RISK"+str(i) for i in cif1.columns]

    deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
    deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
    for i in [1, 3, 5, 7]:
        deep_graft_df["RISK"+str(i)] = cif1["RISK"+str(i)]
    deep_graft_df.loc[deep_graft_df.EVENT==2, "EVENT"] = 0

    avg = tdci_crgf_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")

    print(avg)

    model.save_net(out_file)


def eval_OPTN(OPTN: int, feature:str):
    out_file = f"experiments/results/OPTN_eval/deephit_region-{OPTN}_{feature}.txt"
    with open(out_file, "a") as f:
            f.write(f"{OPTN}\n")

    np.random.seed(1234)
    _ = torch.manual_seed(1234)
    model_file = f"experiments/models/deephit/OPTN/{OPTN}/deepsurv_{feature}_{OPTN}.pkl"

    if OPTN <= 11:
        model = DeepHit(model_file)
    else:
        model = DeepHit(f"experiments/models/deephit/deephit_{feature}_True.pkl")

    for i in range(11):
        OPTN = i+1
        _, val  = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
        _, val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
        _, val_set = load_deephit(feature, OPTN=OPTN)
        x_val, _ = val_set

        cif = model.predict_cif(x_val)
        cif1 = pd.DataFrame(cif[0], model.duration_index).transpose()
        cif1.columns = cif1.columns.astype("int")
        cif1.columns = ["RISK"+str(i) for i in cif1.columns]

        deep_graft_df = val.drop(columns=["EVENT"]).merge(val_sta, on="PX_ID", how="left")
        deep_graft_df = deep_graft_df[["PX_ID", "EVENT", "TIME", "start"]]
        for i in [1, 3, 5, 7]:
            deep_graft_df["RISK"+str(i)] = cif1["RISK"+str(i)]
        deep_graft_df.loc[deep_graft_df.EVENT==2, "EVENT"] = 0

        avg = tdci_crgf_avg("dynamic", deep_graft_df, "RISK", "EVENT", "TIME", "start")

        print(f"OPTN {OPTN}:")
        print(f"tcdi avg: {avg}")
        with open(out_file, "a") as f:
            f.write(f"OPTN {OPTN}: tcdi_avg: {avg:.4f} \n")


if __name__ == "__main__":
    import fire

    fire.Fire()