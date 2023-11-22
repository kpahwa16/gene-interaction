import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        print(f"Initializing MAB with dim_Q: {dim_Q}, dim_K: {dim_K}, dim_V: {dim_V}, num_heads: {num_heads}")
        print(f"fc_k weight shape: {self.fc_k.weight.shape}, fc_v weight shape: {self.fc_v.weight.shape}")
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None, return_attention=False):
      
        print("Entering MAB.forward")
        print(f"Initial Q shape: {Q.shape}, K shape: {K.shape}")
        Q = self.fc_q(Q)

        K, V = self.fc_k(K), self.fc_v(K)
        print(f"After Linear layers Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        print(f"Splitting for heads Q_ shape: {Q_.shape}, K_ shape: {K_.shape}, V_ shape: {V_.shape}")

        attention_scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1,self.num_heads, -1, -1)
#             mask = mask.unsqueeze(1).expand_as(attention_scores)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # Use a large negative number for masked positions

        A = torch.softmax(attention_scores, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if return_attention:
            return O, A.view(Q.size(0), self.num_heads, Q.size(1), K.size(1))
        print("Exiting MAB.forward")
        return O
    
    
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None, return_attention=False):
        print("Entering SAB.forward, Input X shape:", X.shape)
        if return_attention:
            return self.mab(X, X, mask, return_attention=True)
        return self.mab(X, X, mask)
    
# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)

#         # Ensure the dimensions are correct for the MAB modules
#         # dim_in for Q and K should be consistent with the input dimensions
#         self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)  # mab0 processes the learned inducing points I and the input X
#         self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)  # mab1 processes the transformed X (from mab0) and the input X

#         print(f"Inside ISAB: mab0 initialized with dim_Q: {dim_out}, dim_K: {dim_in}, dim_V: {dim_out}, num_heads: {num_heads}")
#         print(f"Inside ISAB: mab1 initialized with dim_Q: {dim_in}, dim_K: {dim_out}, dim_V: {dim_out}, num_heads: {num_heads}")

#     def forward(self, X, mask=None):
#         print("ISAB input X shape:", X.shape)
#         H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
#         print("Output H shape after mab0:", H.shape)
#         output = self.mab1(X, H, mask)
#         print("Exiting ISAB.forward")
#         return output

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        # mab0 processes the learned inducing points I and the input X
        # Both I and X have the same feature dimension (dim_out for I and dim_in for X, which are both 128 in your case)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        
        # mab1 processes the transformed X (from mab0) and the input X
        # Both inputs to mab1 have the same feature dimension (dim_out)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
        output = self.mab1(X, H, mask)
        return output
    
# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)

#         # mab0 processes the learned inducing points I and the input X
#         self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        
#         # mab1 processes the transformed X (from mab0) and the input X
#         self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

#         print(f"Inside ISAB: mab0 initialized with dim_Q: {dim_out}, dim_K: {dim_in}, dim_V: {dim_out}, num_heads: {num_heads}")
#         print(f"Inside ISAB: mab1 initialized with dim_Q: {dim_in}, dim_K: {dim_out}, dim_V: {dim_out}, num_heads: {num_heads}")

#     def forward(self, X, mask=None):
#         print("ISAB input X shape:", X.shape)
#         H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
#         print("Output H shape after mab0:", H.shape)
#         output = self.mab1(X, H, mask)
#         print("Exiting ISAB.forward")
#         return output
# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)
#         print(f"Inside ISAB:dim_out-{dim_out},dim_in-{dim_in},num_heads-{num_heads}")
#         self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
#         self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

#     def forward(self, X, mask=None, return_attention=False):
#         print("ISAB input X shape:", X.shape)
#         print(f"Inside ISAB:Input for mab0-{self.I.repeat(X.size(0), 1, 1).shape,X.shape, mask.shape}")
#         H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
#         print("Output H shape after mab0:", H.shape)
#         if return_attention:
#             output, attention = self.mab1(X, H, mask, return_attention=True)
#             print("Exiting ISAB.forward with attention")
#             return output, attention
#         output = self.mab1(X, H, mask)
#         print("Exiting ISAB.forward")
#         return output
    
# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)
#         # The dim_out of mab0 should match the dim_in of mab1
#         self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
#         self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

#     def forward(self, X, mask=None,return_attention=False):
#         print("ISAB input X shape:", X.shape)
#         H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
#         print("Output H shape after mab0:", H.shape)
#         if return_attention:
#             output, attention = self.mab1(X, H, mask, return_attention=True)
#             print("Exiting ISAB.forward with attention")
#             return output, attention
#         output = self.mab1(X, H, mask)
#         print("Exiting ISAB.forward")
#         return output
# In this corrected version, the key change is in ensuring the dim_in and dim_out parameters are correctly set when initializing the MA

    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None, return_attention=False):
        print("Entering PMA.forward, Input X shape:", X.shape)
        if return_attention:
            return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask, return_attention=True)
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)

