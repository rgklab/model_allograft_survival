import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


def _clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _generate_square_subsequent_mask(sz):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask = torch.triu(torch.full((sz,sz), torch.tensor(-1e9), device=device), diagonal=1)
    return mask


def attention(query, key, value, src_mask=None, pad_mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if pad_mask is not None:
        scores = scores.masked_fill(pad_mask == 0, -1e9)
    # causal attention
    if src_mask is not None:
        scores += src_mask
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert (d_model % h) == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, src_mask, pad_mask=None):
        "Implements Figure 2"
        if pad_mask is not None:
            # Same mask applied to all h heads.
            #mask = mask.unsqueeze(1)
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, src_mask=src_mask, pad_mask=pad_mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


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


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, d_ff, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model, drop_prob)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.sublayer = _clones(SublayerConnection(d_model, drop_prob), 2)

    def forward(self, x, src_mask, pad_mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask, pad_mask))
        return self.sublayer[1](x, self.feed_forward)


class MASFormer(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_features, num_layers: int = 6, d_model: int = 512,
                d_ff: int = 2048, num_heads: int = 8, drop_prob: float=0.1):
        super(MASFormer, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, drop_prob)
        encoder_layer = EncoderLayer(d_model, d_ff, num_heads, drop_prob)
        self.layers = _clones(encoder_layer, num_layers)
        # self.norm = LayerNorm(encoder_layer.size)
        self.final_layer = TranFinalLayer(d_model)
        
    def forward(self, x, pad_mask=None):
        "Pass the input (and mask) through each layer in turn."
        # 20 is the sequence length
        src_mask = _generate_square_subsequent_mask(20)
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, src_mask, pad_mask)
        return self.final_layer(x)


class MASFormer_torch(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_features, num_layers: int = 6, d_model: int = 512,
                d_ff: int = 2048, num_heads: int = 8, drop_prob: float=0.1):
        super(MASFormer_torch, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, drop_prob)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=d_ff, dropout=drop_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_layer = TranFinalLayer(d_model)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        x = self.position_encode(self.src_embed(x))
        # remember the mask here is negated: 1 means to ignore
        x = self.transformer_encoder(x, src_key_padding_mask =mask)
        return self.final_layer(x)