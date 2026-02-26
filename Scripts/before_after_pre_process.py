import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv('tcga_rna_count_data_crc.csv')
df_predic = pd.read_csv('prediction_file_crc.csv')

# 2. Set the first column (Gene ID) as the index
gene_col_df = df.columns[0]
df.set_index(gene_col_df, inplace=True)

num_samples_df = len(df.columns)
num_genes_df = len(df)

#Load Filtered Data
df_filtered = pd.read_csv('filtered_rna_data.csv')

df_predic_filtered = pd.read_csv('filtered_prediction_data.csv')

#set first col as index
gene_col_df_filtered = df_predic.columns[0]
df_predic.set_index(gene_col_df_filtered, inplace=True)

# Create summary for RNA Count Data
count_summary = pd.DataFrame({
    'Before Filtering': [len(df), len(df.columns)],
    'After Filtering': [len(df_filtered), len(df_filtered.columns)]
}, index=['Genes (Rows)', 'Samples (Columns)'])

print("RNA Count Data Summary:")
print(count_summary)

# Create summary for Prediction Data
predic_summary = pd.DataFrame({
    'Before Filtering': [len(df_predic)],
    'After Filtering': [len(df_predic_filtered)]
}, index=['Samples (Rows)'])

print("\nPrediction Data Summary:")
print(predic_summary)