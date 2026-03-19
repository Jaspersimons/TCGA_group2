import pandas as pd
import numpy as np

############# Load Files ##################
# Load RNA expression data and MSI status data
rna_data = pd.read_csv('tcga_rna_count_data_crc.csv')

# Save the gene names
gene_names = rna_data.iloc[:, 0]

prediction_data = pd.read_csv('prediction_file_crc.csv')
###########################################


############## Remove inkown MSI status #################
# Define valid MSI status from the prediction data
valid_msi_status = ['MSI-H', 'MSI-L', 'MSS']

# Filter out rows with not valid MSI status
valid_samples = prediction_data[prediction_data['msi_status'].isin(valid_msi_status)]['Unnamed: 0']

# Filter the RNA data based on the valid samples 
rna_data_filtered = rna_data.loc[:, rna_data.columns.isin(valid_samples)]

# Filter the prediction data (MSI status)
prediction_data_filtered = prediction_data[prediction_data['Unnamed: 0'].isin(valid_samples)]
#############################################################################


############# Zero-heavy genes ##################
# Remove zero-heavy genes (>50%)
zero_threshold = 0.5  

gene_mask = rna_data_filtered.eq(0).mean(axis=1) < zero_threshold
valid_genes = rna_data_filtered[gene_mask]
#################################################


############ Log transformation #################
# Apply log transformation (log(x + 1)) to RNA data
rna_data_log_transformed = np.log1p(valid_genes)  # log1p(x) is log(x + 1)
#################################################


################# Variance filtering ###################
#Calculate the variance for every gene 1
gene_variances = rna_data_log_transformed.var(axis=1)

#75th percentile threshold
top_percentile = np.percentile(gene_variances, 75)

#Keep top 25% most variable genes
filtered_rna_data = rna_data_log_transformed[gene_variances > top_percentile]
#########################################################


############### Re-add Gene Names ########################
filtered_rna_data['Gene_Name'] = gene_names[gene_names.index.isin(filtered_rna_data.index)] 

filtered_rna_data = filtered_rna_data[['Gene_Name'] + [col for col in filtered_rna_data.columns if col != 'Gene_Name']]
#####################################################

############### Save files ########################
filtered_rna_data.to_csv('filtered_rna_data.csv', index=False)

prediction_data_filtered.to_csv('filtered_prediction_data.csv', index=False)
#####################################################




