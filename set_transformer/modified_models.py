import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modified_modules import *

class ModifiedSetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(ModifiedSetTransformer, self).__init__()
        # Define encoder and decoder as individual components
        self.enc1 = ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)
        self.enc2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.dec1 = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dec3 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.final_linear = nn.Linear(dim_hidden, dim_output)
        self.attention_weights = []

    def forward(self, X, mask=None):
        self.attention_weights = []

        # Manually handle sequential processing for encoder
        for layer in self.enc:
            X = layer(X, mask)

        # Manually handle sequential processing for decoder
        dec_out = X
        for layer in self.dec:
            if isinstance(layer, (SAB, PMA)):
                dec_out, attn_weight = layer(dec_out, mask, return_attention=True)
                self.attention_weights.append(attn_weight)
            else:
                dec_out = layer(dec_out)

        return dec_out
        
        
        
