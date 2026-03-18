import pandas as pd
import gseapy as gp
from sklearn.model_selection import train_test_split

############ 1. Load the preprocessed data ##################
# MUST set index_col=0 so Gene IDs and Sample IDs become the dataframe index
rna_data = pd.read_csv('filtered_rna_data.csv', index_col=0)
prediction_data = pd.read_csv('filtered_prediction_data.csv', index_col=0)
#############################################################

################# 2. Align by sample ID #########################
# The index of prediction_data now contains the sample IDs
valid_samples = prediction_data.index

# Align RNA data (filter columns) and prediction data (filter rows)
rna_data_filtered = rna_data.loc[:, rna_data.columns.isin(valid_samples)] 
prediction_data_filtered = prediction_data.loc[prediction_data.index.isin(valid_samples)]

assert rna_data_filtered.shape[1] == prediction_data_filtered.shape[0], "Mismatch in samples"

## Remove Na's
rna_data_filtered = rna_data_filtered[~rna_data_filtered.index.isna()]

rna_data_filtered.index = rna_data_filtered.index.astype(str)
############################################################

################# 3. Run ssGSEA #################################
# Pass the UN-TRANSPOSED data (Genes = Rows, Samples = Columns)
ss_results = gp.ssgsea(
    data=rna_data_filtered, 
    gene_sets='MSigDB_Hallmark_2020', 
    outdir=None, 
    sample_norm_method='rank', 
    no_plot=True
)

# Extract scores (Rows = Pathways, Columns = Samples)
hallmark_scores = ss_results.res2d
print("Hallmark Scores Sample:")
print(hallmark_scores.head())
#################################################################

################ 4. Prepare features for Machine Learning #######
# NOW we transpose the hallmark scores so samples are rows and pathways are columns
X = hallmark_scores.T 
y = prediction_data_filtered['msi_status']

# Split the pathway scores into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#################################################################