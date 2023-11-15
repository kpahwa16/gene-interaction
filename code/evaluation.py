#!/usr/bin/env python
# coding: utf-8

# --method: zirui, guanchu or co_expression or transformer
# --cell: cell type

import gseapy as gp
import pandas as pd
import numpy as np
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--cell', required=True)
    return parser

# Generate custom gene set
def custom_gene_set(file):
    df_gene_set = pd.read_csv(file)
    gene_set = df_gene_set.to_dict(orient='list')
    return gene_set

# Generate ranking list
def ranking_list(file):
    ranking = pd.read_csv(file)
    ranking = ranking.sort_values('avg_score', ascending=False)
    ranking = ranking.reset_index()
    df_sort = pd.DataFrame(np.sort(ranking[['gene_a', 'gene_b']], axis=1)) # sort gene_a and gene_b
    del ranking
    rnk = pd.DataFrame({'0':(df_sort[0]+','+df_sort[1]).tolist(),
                        '1':list(reversed(list(range(df_sort.shape[0]))))})
    rnk = rnk.set_index('0')
    del df_sort
    return rnk

# Run GSEA
def run_gsea(path_data, method, cell, file, gene_set):   
    rnk = ranking_list(os.path.join(path_data, 'random_samples', method, cell, file))
    l_random_samples = []
    l_gene_set = []
    l_ES = []
    l_NES = []
    l_p_value = []
    l_tag = []
    l_gene = []
    pre_res = gp.prerank(rnk=rnk,
                         gene_sets=gene_set,
                         min_size=1,
                         max_size=1000000000,
                         permutation_num=1000, 
                         outdir=None,
                         seed=6,
                         verbose=False)
    terms = pre_res.res2d.Term
    for i in range(3):
        l_random_samples.append(file.replace('.csv', ''))
        l_gene_set.append(terms[i])
        l_ES.append(pre_res.results[terms[i]]['es'])
        l_NES.append(pre_res.results[terms[i]]['nes'])
        if pre_res.results[terms[i]]['pval'] == 0:
            l_p_value.append(0.001) # min_p_value = 1/number of permutations
        else:
            l_p_value.append(pre_res.results[terms[i]]['pval'])   
        l_tag.append(pre_res.results[terms[i]]['tag %'])
        l_gene.append(pre_res.results[terms[i]]['gene %'])
    res_gsea = pd.DataFrame({'random_samples': l_random_samples,
                             'gene_set':l_gene_set,
                             'ES':l_ES,
                             'NES':l_NES,
                             'p_value':l_p_value,
                             'tag %':l_tag,
                             'gene %':l_gene})
    return res_gsea

def main():
    # Set up parameters
    parser = get_parser()
    args = parser.parse_args()
    method = args.method
    cell = args.cell
    path_data = 'SEA_AD_project/'
    # Load data
    BioGRID = custom_gene_set(os.path.join(path_data, 'raw_data', 'geneset', 'BioGRID.csv'))
    DisGeNET_All = custom_gene_set(os.path.join(path_data, 'raw_data', 'geneset', 'DisGeNET_All.csv'))
    DisGeNET_Strong = custom_gene_set(os.path.join(path_data, 'raw_data', 'geneset', 'DisGeNET_Strong.csv'))
    gene_set = {**BioGRID, **DisGeNET_All, **DisGeNET_Strong}
    # GSEA
    df_gsea = pd.DataFrame()
    for file in os.listdir(os.path.join(path_data, 'random_samples', method, cell)):
        res_gsea = run_gsea(path_data, method, cell, file, gene_set)
        df_gsea = pd.concat([df_gsea, res_gsea], ignore_index=True)
    df_gsea.to_csv(os.path.join(path_data, 'gsea', method, cell+'.csv'), index=False)

if __name__ == "__main__":
    main()
