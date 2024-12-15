import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# final layer for the transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        return torch.sigmoid(x.squeeze(-1))


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super(GRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # GRU layers
        self.GRU = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )

        self.final_layer = TranFinalLayer(hidden_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.GRU(x, h0.detach())

        out = self.final_layer(out)

        return out