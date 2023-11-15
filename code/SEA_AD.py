import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdf5plugin
import os

path_data = '/data/kpahwa/gene-interact/RNAseq/'

data = sc.read_h5ad(os.path.join(path_data,
                                 'SEAAD_MTG_RNAseq_final-nuclei.2022-08-18.h5ad'))

cell_information = data.obs
cell_information = cell_information[cell_information['Overall AD neuropathological Change']!='Reference']
print(cell_information.shape)
cell_information.to_csv(os.path.join(path_data, 'out', 'cell_information.csv'),
                                     index=False)

cell_counts = cell_information[['Overall AD neuropathological Change','Subclass']].value_counts().rename_axis(['Overall AD neuropathological Change','Subclass']).reset_index(name='counts')
cell_counts = cell_counts.sort_values('Subclass')
cell_counts.to_csv(os.path.join(path_data, 'out', 'cell_count.csv'),
                                index=False)
gene_list = data.var
gene_list.to_csv(os.path.join(path_data, 'out', 'gene_list.csv'),
                              index=False)

expression_matrix = pd.DataFrame(data.X.toarray(), index=data.obs_names)
expression_matrix.columns = data.var_names
expression_matrix = expression_matrix[expression_matrix.index.isin(cell_information['sample_id'].to_list())]

print(expression_matrix.shape)
# Save the gene expression data of each cell type separately
l_cell_type = []
l_data = []
n = 0
for name, group in cell_information[['Overall AD neuropathological Change', 'Subclass']].groupby('Subclass'):
    matrix = expression_matrix[expression_matrix.index.isin(group.index.to_list())]
    adata = sc.AnnData(X = matrix.to_numpy(), var = gene_list, obs = group)
    adata.write_h5ad(os.path.join(path_data, 'out', 'gene_expression',
                                  'cell_type_'+str(n)+'.h5ad'),
                     compression=hdf5plugin.FILTERS["zstd"],
                     compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)
    l_cell_type.append(name)
    l_data.append('cell_type_'+str(n)+'.h5ad')
    n += 1
