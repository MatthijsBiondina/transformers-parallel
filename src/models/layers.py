import torch.nn as nn

from src.models.sublayers import FeedForward, MultiHeadAttention, Norm
from src.utils.tools import Tools as T


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm_1(x)
        x = x + self.dropout_1(self.attn(h, h, h, mask))
        h = self.norm_2(x)
        x = x + self.dropout_2(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):

        h = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(h, h, h, trg_mask))
        h = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(h, e_outputs, e_outputs, src_mask))
        h = self.norm_3(x)
        x = x + self.dropout_3(self.ff(h))
        return x
