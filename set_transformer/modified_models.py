import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from modified_modules import *

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
        print("Input X shape:", X.shape)
        print("Input mask shape:", mask.shape if mask is not None else "No mask")
        self.attention_weights = []

        # Process through the first encoder layer
        X = self.enc1(X, mask)
        print("After enc1 X shape:", X.shape)
        # Process through the second encoder layer
        X = self.enc2(X, mask)
        print("After enc2 X shape:", X.shape)
        # Initialize decoder output
        dec_out = X
        print("dec_out initialized shape:", dec_out.shape)

        # Process through the first decoder layer
        dec_out, attn_weight = self.dec1(dec_out, mask, return_attention=True)
        print("After dec1 dec_out shape:", dec_out.shape)
        self.attention_weights.append(attn_weight)

        # Process through the second decoder layer
        dec_out, attn_weight = self.dec2(dec_out, mask, return_attention=True)
        print("After dec2 dec_out shape:", dec_out.shape)
        self.attention_weights.append(attn_weight)

        # Process through the third decoder layer
        dec_out, attn_weight = self.dec3(dec_out, mask, return_attention=True)
        print("After dec3 dec_out shape:", dec_out.shape)
        self.attention_weights.append(attn_weight)

        # Apply the final linear layer
        dec_out = self.final_linear(dec_out)
        print("After final_linear dec_out shape:", dec_out.shape)

        return dec_out
        

