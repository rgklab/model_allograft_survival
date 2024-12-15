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


def train(feature:str, normalized: bool = True):

    wandb.init(project="test", entity="xgao")
    wandb.define_metric("tdci", summary="max")
    # hparams = dict(wandb.config)

    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    train, val, _  = load_dcox_gfcr()
    _, val_sta, _ = load_SRTR_static_gfcr()
    train, val_set, _ = load_deephit(feature, normalized=normalized)
    x_train, y_train = train
    x_val, _ = val_set

    in_features = x_train.shape[1]
    num_risks = y_train[1].max()
    cuts = np.arange(20, dtype="float64")
    out_features = 20

    epochs = 512
    batch_norm = bool(wandb.config.batch_norm)
    dropout = wandb.config.dropout
    num_nodes_shared = wandb.config.nodes_shared
    num_layers_shared = wandb.config.layers_shared
    num_nodes_indiv = wandb.config.nodes_indiv
    num_layers_indiv = wandb.config.layers_indiv
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    alpha = wandb.config.alpha
    sigma = wandb.config.sigma
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

    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val_set)

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

    metrics = {"tdci": avg}
    wandb.log(metrics)


def train_OPTN(OPTN: int, feature:str, normalized: bool = True):
    wandb.init(project="test", entity="xgao")
    wandb.define_metric("tdci", summary="max")
    # hparams = dict(wandb.config)

    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    train, val  = load_OPTN_crgf(OPTN=OPTN, mode="dynamic")
    _, val_sta = load_OPTN_crgf(OPTN=OPTN, mode="static")
    train, val_set = load_deephit(feature, normalized=normalized, OPTN=OPTN)
    x_train, y_train = train
    x_val, _ = val_set

    in_features = x_train.shape[1]
    num_risks = y_train[1].max()
    cuts = np.arange(20, dtype="float64")
    out_features = 20

    epochs = 512
    batch_norm = bool(wandb.config.batch_norm)
    dropout = wandb.config.dropout
    num_nodes_shared = wandb.config.nodes_shared
    num_layers_shared = wandb.config.layers_shared
    num_nodes_indiv = wandb.config.nodes_indiv
    num_layers_indiv = wandb.config.layers_indiv
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    alpha = wandb.config.alpha
    sigma = wandb.config.sigma
    layer_shared = [num_nodes_shared for i in range(num_layers_shared)]
    layer_indiv = [num_nodes_indiv for i in range(num_layers_indiv)]

    net = CauseSpecificNet(in_features, layer_shared, layer_indiv, num_risks,
                        out_features, batch_norm, dropout)
    callbacks = [tt.callbacks.EarlyStoppingCycle()]
    verbose = False

    optimizer = tt.optim.AdamWR(lr=lr, decoupled_weight_decay=0.01,
                            cycle_eta_multiplier=0.8)
    model = DeepHit(net, optimizer, alpha=alpha, sigma=sigma,
                    duration_index=cuts)

    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val_set)

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

    metrics = {"tdci": avg}
    wandb.log(metrics)


def run_agent(sweep_id: str, project_name: str, count: int = 1, feature: str = "mas", OPTN = None):

    if OPTN:
        train_func = partial(train_OPTN, OPTN=OPTN, feature=feature, normalized=True)
    else:
        train_func = partial(train, feature=feature, normalized=True)

    wandb.agent(sweep_id=sweep_id,
                count=count,
                project= project_name,
                function=train_func)


def final_model(feature:str, normalized: bool = True):

    # full model: 0.856
    # mas model: 0.8482

    np.random.seed(1234)
    _ = torch.manual_seed(1234)
    out_file = f"experiments/models/deephit/deephit_{feature}_{normalized}.pkl"

    if feature == "mas":
        hparams = {'alpha': 0.5, 'batch_norm': 1, 'batch_size': 256, 'dropout': 0, 'layers_indiv': 1, 'layers_shared': 2,
                    'lr': 0.01, 'nodes_indiv': 128, 'nodes_shared': 128, 'sigma': 0.1}
    else:
        hparams = {'alpha': 0.2, 'batch_norm': 1, 'batch_size': 256, 'dropout': 0.5, 'layers_indiv': 1, 'layers_shared': 2,
                    'lr': 0.005, 'nodes_indiv': 16, 'nodes_shared': 16, 'sigma': 0.1}

    train, val, _  = load_dcox_gfcr()
    _, val_sta, _ = load_SRTR_static_gfcr()
    train, val_set, _ = load_deephit(feature, normalized=normalized)
    x_train, y_train = train
    x_val, _ = val_set

    in_features = x_train.shape[1]
    num_risks = y_train[1].max()
    cuts = np.arange(20, dtype="float64")
    out_features = 20

    epochs = 512
    batch_norm = bool(hparams["batch_norm"])
    dropout = hparams["dropout"]
    num_nodes_shared = hparams["nodes_shared"]
    num_layers_shared = hparams["layers_shared"]
    num_nodes_indiv = hparams["nodes_indiv"]
    num_layers_indiv = hparams["layers_indiv"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    alpha = hparams["alpha"]
    sigma = hparams["sigma"]
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

    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val_set)

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



if __name__ == "__main__":
    import fire

    fire.Fire()