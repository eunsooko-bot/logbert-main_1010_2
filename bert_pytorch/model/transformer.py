import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F

from .utils.gelu import GELU

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .utils.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.norm = LayerNorm(hidden)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.w_1 = nn.Linear(hidden, feed_forward_hidden)
        self.w_2 = nn.Linear(feed_forward_hidden, hidden)
        self.activation = GELU()

    def forward(self, x, mask):
        src = self.norm(x)
        src2 = src
        src2 = self.attention(src2, src2, src2, mask=mask)
        src2 = src2[0]
        src2 = src2 * self.resweight
        src =  src + self.dropout(src2)

        src2 = src
        a=self.w_1(src2)
        b=self.activation(a)
        c=self.dropout(b)
        src2=self.w_2(c)
#        src2 = self.w_2(self.dropout(self.activation(self.w_1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout(src2)
        return src

