import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import os
import argparse
from models import SetTransformer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from modified_models import *
from modified_modules import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--h5ad_file', type=str, help='Path to h5ad file for a specific cell type')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--model_save_path', type=str, default='./set_transformer_model.pth')
args = parser.parse_args()

# Load data
adata = sc.read_h5ad(args.h5ad_file)

# Extract gene expression matrix and labels
X = adata.X
y = LabelEncoder().fit_transform(adata.obs['Overall AD neuropathological Change'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class GeneExpressionDataset(Dataset):
    def __init__(self, expressions, labels):
        self.expressions = expressions
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.expressions[idx].toarray(), dtype=torch.float), self.labels[idx]

# Datasets and Dataloaders
train_dataset = GeneExpressionDataset(X_train, y_train)
test_dataset = GeneExpressionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Model
class GeneInteractionModel(nn.Module):
    def __init__(self, num_genes, num_classes):
        super(GeneInteractionModel, self).__init__()
        self.set_transformer = ModifiedSetTransformer(dim_input=num_genes, num_outputs=num_classes, dim_output=num_classes)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.set_transformer(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

model = GeneInteractionModel(num_genes=X.shape[1], num_classes=len(np.unique(y))).cuda()
# Training
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), args.model_save_path)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy: {} %'.format(100 * correct / total))

# analysing attention weights

all_attention_weights = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.cuda()
        _ = model(data)
        all_attention_weights.append(model.attention_weights)

aggregate_attention = torch.mean(torch.stack(all_attention_weights), dim=[0, 1])

# Flatten and sort the attention matrix to rank gene interactions
flattened_attention = aggregate_attention.flatten()
sorted_indices = torch.argsort(flattened_attention, descending=True)

# providing gene_names in a sorted list
ranked_interactions = [(gene_names[i // num_genes], gene_names[i % num_genes], flattened_attention[i].item()) 
                       for i in sorted_indices]

# Print top interactions
for interaction in ranked_interactions[:10]:
    print(interaction)