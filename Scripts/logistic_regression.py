import pandas as pd
import gseapy as gp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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


rna_data_filtered.index = [str(idx) for idx in rna_data_filtered.index]

# 2. Drop any rows where the index evaluated to the string 'nan'
rna_data_filtered = rna_data_filtered[rna_data_filtered.index != 'nan']

# 3. Drop duplicate gene names (keeps the first occurrence), which gseapy requires
rna_data_filtered = rna_data_filtered[~rna_data_filtered.index.duplicated(keep='first')]


############################################################

################# 3. Run ssGSEA #################################
# 4. Run ssGSEA
#ss_results = gp.ssgsea(data=rna_data_filtered, gene_sets='MSigDB_Hallmark_2020', outdir=None, sample_norm_method='rank', no_plot=True)

# Extract scores (Rows = Pathways, Columns = Samples)
#hallmark_scores = ss_results.res2d
#print("Hallmark Scores Sample:")
#print(hallmark_scores.head())
#################################################################

################ 4. Prepare features for Machine Learning #######
# NOW we transpose the hallmark scores so samples are rows and pathways are columns
#X = hallmark_scores.T 
y = prediction_data_filtered['msi_status']

X = rna_data_filtered.T


# Split the pathway scores into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#################################################################



############################# Standardize Data ############################
# Logistic Regression with L1/Elastic Net requires scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
###########################################################################

###########################################################################
# OPTION A: 3-Class Model (MSI-H, MSI-L, MSS)
###########################################################################
print("--- 3-Class Model ---")
log_reg_3class = LogisticRegression(
    penalty='l1', 
    solver='saga', 
    max_iter=5000,
    class_weight='balanced', 
    random_state=42
)
log_reg_3class.fit(X_train_scaled, y_train)

y_pred_3class = log_reg_3class.predict(X_test_scaled)
print(classification_report(y_test, y_pred_3class))


###########################################################################
# OPTION B: 2-Class Binary Model (MSI-L vs MSS)
###########################################################################
print("\n--- 2-Class Model (MSI-L vs MSS) ---")

# 1. Filter the training and testing sets to only include MSI-L and MSS
train_mask = y_train.isin(['MSI-L', 'MSS'])
test_mask = y_test.isin(['MSI-L', 'MSS'])

X_train_scaled_bin = X_train_scaled[train_mask]
y_train_bin = y_train[train_mask]

X_test_scaled_bin = X_test_scaled[test_mask]
y_test_bin = y_test[test_mask]

# 2. Train the binary model
log_reg_bin = LogisticRegression(
    penalty='l1', 
    solver='saga', 
    max_iter=5000, 
    class_weight='balanced', 
    random_state=42
)
log_reg_bin.fit(X_train_scaled_bin, y_train_bin)

# 3. Predict and evaluate
y_pred_bin = log_reg_bin.predict(X_test_scaled_bin)
print(classification_report(y_test_bin, y_pred_bin))