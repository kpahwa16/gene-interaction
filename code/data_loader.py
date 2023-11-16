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
