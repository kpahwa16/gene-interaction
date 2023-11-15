import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdf5plugin
import scanpy as sc
import numpy as np
import os

# Setting the path
path_data = '/data/kpahwa/gene-interact/RNAseq/'

# Loading the data
data_cell_type_9 = sc.read_h5ad(os.path.join(path_data, "out", "gene_expression", 'cell_type_9.h5ad'))

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

#Visualizing Expression Data of Top Genes
sc.pl.highest_expr_genes(data_cell_type_0, n_top=20, show=True)

#Computing UMAP (this assumes you've preprocessed the data)
sc.tl.umap(data_cell_type_0)

#Visualizing using UMAP with categorical metadata as color
sc.pl.umap(data_cell_type_0, color=['Overall AD neuropathological Change', 'Subclass'], show=True)

