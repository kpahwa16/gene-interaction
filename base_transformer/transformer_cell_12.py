import wandb
wandb.login(key="1c8e6b0c37578706a3a233c97a56844b2281bf21")
wandb.init(project='gene-interaction-cell_type-12', entity='khushbu16pahwa')
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import os
import argparse
# from models import SetTransformer
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdf5plugin
import scanpy as sc
import numpy as np
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
# Setting the path
path_data = '/data/kpahwa/gene-interact/RNAseq/'
cell_type = "12"
# Loading the data
data_cell_type = sc.read_h5ad(os.path.join(path_data, "out", "gene_expression", 'cell_type_12.h5ad'))
# # Loading the data
#data_cell_type = sc.read_h5ad("/data/kpahwa/gene-interact/RNAseq/out/gene_expresssion/cell_type_9.h5ad")
# data_cell_type_11 = sc.read_h5ad(os.path.join(path_data, "out", "gene_expression", 'cell_type_11.h5ad'))

# Printing basic information
print(data_cell_type)

# Inspecting the .obs attribute
print("\nObservations (obs) metadata:")
print(data_cell_type.obs.head())

# Inspecting the .var attribute
print("\nVariables (var) metadata:")
print(data_cell_type.var.head())

# Summary of the data
print(f'\nDataset shape: {data_cell_type.shape}')

# Checking Missing Values
print(f'\nNumber of missing values: {np.isnan(data_cell_type.X).sum()}')

# Exploring Unique Values in Categorical Metadata
print("\nUnique values in 'Overall AD neuropathological Change':")
print(data_cell_type.obs['Overall AD neuropathological Change'].value_counts())

print("\nUnique values in 'Subclass':")
print(data_cell_type.obs['Subclass'].value_counts())


X = data_cell_type.X  # Sparse Matrix

filtered_data = data_cell_type[data_cell_type.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
X_filtered = filtered_data.X  # Your filtered gene expression data
y_filtered = filtered_data.obs['Overall AD neuropathological Change']  # Your filtered labels
label_mapping = {'Not AD': 0, 'High': 1}
y_encoded = y_filtered.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # Adjust test_size as needed
X_train_sparse = csr_matrix(X_train)
X_test_sparse = csr_matrix(X_test)
X_val_sparse = csr_matrix(X_val)




class VariableLengthSequenceDataset(Dataset):
    def __init__(self, sequences, labels, scaling_factors):
        self.sequences = sequences
        self.labels = labels
        self.scaling_factors = scaling_factors  # Add scaling_factors

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        scaling_factor = self.scaling_factors[index]  # Get scaling_factor
        return sequence, label, scaling_factor  # Return sequence, label, and scaling_factor

def custom_collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences, labels, scaling_factors = [], [], []

    for item in batch:
        sequences.append(item[0])
        labels.append(item[1])
        scaling_factors.append(item[2])

    # Create a tensor for sequences with padding
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    # Create a tensor for labels
    labels_tensor = torch.tensor(labels)

    # Create a tensor for scaling_factors with padding
    scaling_factors_padded = torch.nn.utils.rnn.pad_sequence(scaling_factors, batch_first=True, padding_value=0)

    # Create a mask to identify padding elements
    mask = (sequences_padded != 0)
  
    return sequences_padded, labels_tensor, mask, scaling_factors_padded



train_sequences = []
train_scaling_factors = []  # To store scaling factors

for i in range(X_train_sparse.shape[0]):
    x = X_train_sparse[i].toarray()
    x_tensor = torch.from_numpy(x).float()  # Ensure it's a float tensor for the following operations
    non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
    train_sequences.append(non_zero_indices)
    
    scaling_factor = x_tensor[:,non_zero_indices]
    train_scaling_factors.append(scaling_factor)

val_sequences = []
val_scaling_factors = []  # To store scaling factors

for i in range(X_val_sparse.shape[0]):
    x = X_val_sparse[i].toarray()
    x_tensor = torch.from_numpy(x).float()  # Ensure it's a float tensor for the following operations
    non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
    val_sequences.append(non_zero_indices)
    
    scaling_factor = x_tensor[:,non_zero_indices]
    val_scaling_factors.append(scaling_factor)

test_sequences = []
test_scaling_factors = []  # To store scaling factors

for i in range(X_test_sparse.shape[0]):
    x = X_test_sparse[i].toarray()
    x_tensor = torch.from_numpy(x).float()  # Ensure it's a float tensor for the following operations
    non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
    test_sequences.append(non_zero_indices)

    scaling_factor = x_tensor[:,non_zero_indices]
    test_scaling_factors.append(scaling_factor)

print(len(train_scaling_factors), len(test_scaling_factors))
print(train_scaling_factors[0], test_scaling_factors[0])


# modeling :


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         print("inside forward:mask.shape", mask.shape)
     
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         print(dots.shape)
        # Apply masking
#         print("inside forward:dots.shape before masking", dots.shape)
        if mask is not None:
            large_negative = -1e9
            dots = dots.masked_fill(~mask.unsqueeze(1).unsqueeze(2),large_negative)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # Mask the output
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(2), 0)

          # Apply scaling factors

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class MultiLayerModel(nn.Module):
    def __init__(self, num_genes, embedding_dim, depth=4, heads=8, dim_head=64, mlp_dim=64, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(num_genes, embedding_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embedding_dim)
        # Create a list to hold attention and feedforward layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        Attention(embedding_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(embedding_dim, mlp_dim, dropout=dropout)
                    ]))
        self.output_linear = nn.Linear(embedding_dim, 2)

    def forward(self, x, mask, scaling_factors):
        x = self.embedding(x)
        x = x * scaling_factors.unsqueeze(2)  # Broadcasting scaling_factors
        
        #intermediate_x = x.clone()  # Create a clone of x to hold the intermediate values
        for attn, ff in self.layers:
            x = attn(x,mask) + x
            x = ff(x) + x
            
        x = self.norm(x) 
        # Pool over the sequence dimension (you can change this as needed)
        x_pooled = torch.mean(x, dim=1)
        out = self.output_linear(x_pooled)
        return out
    


# preparing data to pass to the dataloader:
train_scaling_factors = list(x[0] for x in train_scaling_factors)
test_scaling_factors = list(x[0] for x in test_scaling_factors)
val_scaling_factors = list(x[0] for x in val_scaling_factors)


train_labels = list(y_train)
val_labels = list(y_val)
test_labels = list(y_test)

train_dataset = VariableLengthSequenceDataset(train_sequences, train_labels,train_scaling_factors)
# train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=custom_collate_fn, shuffle = True)
    # Added multiple workers for parallel data loading
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=custom_collate_fn, num_workers=4, shuffle = True)
test_dataset = VariableLengthSequenceDataset(test_sequences,test_labels, test_scaling_factors)
#test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=custom_collate_fn, shuffle = False)
    # Added multiple workers for parallel data loading
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=custom_collate_fn, num_workers=4,shuffle = False)
val_dataset = VariableLengthSequenceDataset(val_sequences, val_labels, val_scaling_factors)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate_fn, shuffle=False)

# specifying num_genes and embedding_dim
num_genes = X_train_sparse.shape[1]
embedding_dim = 512
heads = 8
dim_head = 64
mlp_dim = 2048
dropout = 0.0
#just for checking
    # Logging hyperparameters with Weights & Biases
wandb.config.update({
        'embedding_dim': embedding_dim,
        'heads': heads,
        'dim_head': dim_head,
        'mlp_dim': mlp_dim,
        'dropout': dropout
    })

# 
device_ids = [0,1,2,3,4,5,6,7]
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
model = MultiLayerModel(num_genes, embedding_dim=embedding_dim, heads=heads, dim_head=dim_head, mlp_dim= mlp_dim, dropout=dropout)
model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

# initializing model weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)


# starting training and evaluation 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
#Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Training and evaluation
early_stopping = True
patience = 5  # Number of epochs with no improvement on test accuracy before stopping
no_improvement_count = 0
# Training loop

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch_sequences, batch_labels, mask, scaling_factors_batch in dataloader:
        # Move tensors to GPU
        batch_sequences = batch_sequences.to(device)
#         print("batch sequences.shape inside train", batch_sequences.shape)
        batch_labels = batch_labels.to(device)
        print("Batch labels:",batch_labels)
#         print(batch_labels.shape)
        mask = mask.to(device)
#         print("mask inside train: mask.shape", mask.shape)
        scaling_factors_batch = scaling_factors_batch.to(device)

        optimizer.zero_grad()
        output = model(batch_sequences, mask, scaling_factors_batch)
        _, predicted = torch.max(output.data, 1)
        print("Predicted labels:", predicted)
        print("-"*80)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation loop

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_sequences, batch_labels, mask, scaling_factors_batch in dataloader:
            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            print("Batch labels:",batch_labels)
            mask = mask.to(device)
#             print(mask.shape)
            scaling_factors_batch = scaling_factors_batch.to(device)

            output = model(batch_sequences, mask, scaling_factors_batch)
#             output =  model({'x': batch_sequences, 'mask': mask, 'scaling_factors': scaling_factors_batch})
            probabilities = torch.softmax(output, dim=1)

            # Get the predicted class (class with the highest probability)
            predicted_classes = torch.argmax(probabilities, dim=1)


            loss = criterion(output, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            print("Predicted labels:", predicted)
            print("-"*80)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    f1 = f1_score(y_true, y_pred, average='weighted') * 100        
    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy, f1

# Training and evaluation
num_epochs = 20
best_val_accuracy = 0.0  # Change to validation accuracy

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader, criterion)  # Use validation dataloader
    test_loss, test_accuracy, test_f1 = evaluate(model, test_dataloader, criterion)  # Keep for final test evaluation

    # Update learning rate scheduler
    scheduler.step(val_loss)  # Use validation loss

    # Logging metrics with Weights & Biases
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    })

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.2f}%")
    print(f"  Validation F1 score: {val_f1:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Test F1 score: {test_f1:.2f}%")

    if (val_accuracy > best_val_accuracy):
        best_val_accuracy = val_accuracy
        best_model_path = f"/data/kpahwa/gene-interact/code/best_models/best_model_cell_type_{cell_type}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"  Best Model saved to {best_model_path} with validation accuracy {best_val_accuracy:.2f}% and f1 score: {val_f1:.2f}%")
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if early_stopping and no_improvement_count >= patience:
        print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation accuracy.")
        break

# Closing the Weights & Biases run
wandb.finish()
