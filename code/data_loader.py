import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

class GeneExpressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_data_loaders(gex, labels, batch_size=64, train_split=0.8):
    # Filter out classes and encode labels
    binary_labels = np.array(labels)[np.isin(labels, ['High', 'Not AD'])]
    encoder = LabelEncoder()
    binary_labels_encoded = encoder.fit_transform(binary_labels)  # Encodes 'High' and 'Not AD' to 0 and 1

    # Filter gene expression data accordingly
    filtered_gex = gex[np.isin(labels, ['High', 'Not AD']), :]

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(filtered_gex, dtype=torch.float32)
    labels_tensor = torch.tensor(binary_labels_encoded, dtype=torch.long)

    # Create dataset
    dataset = GeneExpressionDataset(features_tensor, labels_tensor)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    path_data = '/data/kpahwa/gene-interact/RNAseq/'
    
    # Loading the data
    data_cell_type_9 = sc.read_h5ad(os.path.join(path_data, "out", "gene_expression", 'cell_type_9.h5ad'))
    # data_cell_type_11 = sc.read_h5ad(os.path.join(path_data, "out", "gene_expression", 'cell_type_11.h5ad'))
    
    # Printing basic information
    print(data_cell_type_9)
    
    # Inspecting the .obs attribute
    print("\nObservations (obs) metadata:")
    print(data_cell_type_9.obs.head())
    
    # Inspecting the .var attribute
    print("\nVariables (var) metadata:")
    print(data_cell_type_9.var.head())
    
    # Summary of the data
    print(f'\nDataset shape: {data_cell_type_9.shape}')
    
    # Checking Missing Values
    print(f'\nNumber of missing values: {np.isnan(data_cell_type_9.X).sum()}')
    
    # Exploring Unique Values in Categorical Metadata
    print("\nUnique values in 'Overall AD neuropathological Change':")
    print(data_cell_type_9.obs['Overall AD neuropathological Change'].value_counts())
    
    print("\nUnique values in 'Subclass':")
    print(data_cell_type_9.obs['Subclass'].value_counts())
    
    labels = (data_cell_type_9.obs["Overall AD neuropathological Change"].tolist())
    gex = data_cell_type_9.X
    create_data_loaders(gex, labels, batch_size=64, train_split=0.8)
    
