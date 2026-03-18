import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

#################### Load data #########################################
rna_data = pd.read_csv('tcga_rna_count_data_crc.csv', index_col='Gene_Name')

prediction_data = pd.read_csv('prediction_file_crc.csv').set_index('Unnamed: 0')
#############################################################################

####################### Couple files and data cleaning ########################
common_samples = rna_data.columns.intersection(prediction_data.index)
rna_subset = rna_data[common_samples].T
prediction_subset = prediction_data.loc[common_samples].copy()

# delete empty and indeterminate
prediction_subset['msi_status'] = prediction_subset['msi_status'].astype(str).str.strip()
valid_statuses = ['MSS', 'MSI-H', 'MSI-L']
mask = prediction_subset['msi_status'].isin(valid_statuses)
prediction_subset = prediction_subset[mask]
rna_subset = rna_subset.loc[prediction_subset.index]

# Log transform
rna_subset = np.log2(rna_subset + 1)
###############################################################################


###################### Feature selection ##################################
target_genes = ['APC', 'BRAF', 'TP53', 'TTN']
X_rna = rna_subset[target_genes]
X_mut = prediction_subset[target_genes].replace({'WT': 0, 'SNV': 1})
X_mut.columns = [f"{g}_mut" for g in target_genes]

X = pd.concat([X_rna, X_mut], axis=1)
y = prediction_subset['msi_status'].map({'MSS': 0, 'MSI-H': 1, 'MSI-L': 2})

# train/test split
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
##############################################################################


############### undersampling ################################
rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority')
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_raw, y_train_raw)
############################################################


#################### Gene weights ########################################
def calculate_custom_weights(X_df, y_series):
    weights = np.ones(len(y_series))
    for i, (idx, row) in enumerate(X_df.iterrows()):
        status = y_series.iloc[i]
        # BRAF/TTN mutation -> MSI-H (1)
        if (row['BRAF_mut'] == 1 or row['TTN_mut'] == 1) and status == 1:
            weights[i] *= 2.5
        # APC/TP53 mutation -> MSS (0) or MSI-L (2)
        if (row['APC_mut'] == 1 or row['TP53_mut'] == 1) and status in [0, 2]:
            weights[i] *= 2.5
    return weights

sample_weights = calculate_custom_weights(X_train_resampled, y_train_resampled)
########################################################################


######################## model training ################################
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', None]
}

rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=cv, 
    scoring='f1_macro', 
    n_jobs=-1, 
    random_state=42
)

random_search.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
best_rf = random_search.best_estimator_
############################################################################


###################### prediction and evaluation #########################
y_pred = best_rf.predict(X_test)
target_names = ['MSS', 'MSI-H', 'MSI-L']

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
########################################################################


#################### Visualisation #####################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=target_names, yticklabels=target_names)
ax1.set_title('Confusion Matrix (RNA + Mutations)')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Classification Report Bar plot
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :3].T
report_df.plot(kind='bar', ax=ax2)
ax2.set_title('Metrics per Class (F1, Precision, Recall)')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('analysis_4_gene_isolation.png', dpi=300)
plt.show()
###########################################################################