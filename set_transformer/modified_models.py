import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modified_modules import *

class ModifiedSetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(ModifiedSetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))
        self.attention_weights = []

    def forward(self, X, mask=None):
        self.attention_weights = []
        # Passing the mask through the encoder and decoder
        enc_out = self.enc(X, mask)
        dec_out, attn_weight = self.dec(enc_out, mask, return_attention=True)
        self.attention_weights.append(attn_weight)
        return dec_out
