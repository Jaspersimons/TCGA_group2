import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import randint
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

############## Configure colors##################
metric_color = ['#7fb3d5', '#f7dc6f', '#27ae60'] # Precision, Recall, F1-score
###########################################################


################ load data ##########################
rna_data = pd.read_csv('filtered_rna_data.csv', index_col=0)
prediction_data = pd.read_csv('filtered_prediction_data.csv').set_index('Unnamed: 0')

common_ids = rna_data.columns.intersection(prediction_data.index)
X_all = rna_data[common_ids].T
y_all = prediction_data.loc[common_ids, 'msi_status']
######################################################


#################### cross validation settings ######################
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2']
}
####################################################################


############# Analysis 1: Binary test = MSI-L ######################
print("\n--- Analysis 1: MSI-L as test ---")
train_mask_1 = y_all.isin(['MSI-H', 'MSS'])
X_train_1 = X_all[train_mask_1]
y_train_1 = y_all[train_mask_1].map({'MSI-H': 1, 'MSS': 0})
X_test_1 = X_all[y_all == 'MSI-L']
y_test_1 = pd.Series(0, index=X_test_1.index) # label MSI-L as 0 (Non-MSI-H)

#set seed
rus = RandomUnderSampler(random_state=42)
X_res_1, y_res_1 = rus.fit_resample(X_train_1, y_train_1)

rs_1 = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=cv_strategy, scoring='accuracy', n_jobs=-1, random_state=42)
rs_1.fit(X_res_1, y_res_1)
y_pred_1 = rs_1.predict(X_test_1)

# Visualisation analysis 1
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cm1 = confusion_matrix(y_test_1, y_pred_1, labels=[0, 1])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Purples', ax=ax1, xticklabels=['Non-MSI-H', 'MSI-H'], yticklabels=['Non-MSI-H', 'MSI-H'])
ax1.set_title('Confusion Matrix (MSI-L Test Set)')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

report1 = classification_report(y_test_1, y_pred_1, target_names=['Non-MSI-H', 'MSI-H'], output_dict=True, zero_division=0)
pd.DataFrame(report1).iloc[:-1, :2].T.plot(kind='bar', ax=ax2, color=metric_color)
ax2.set_title('Classification Metrics')
plt.suptitle('Binary test on MSI-L samples')
plt.tight_layout()
plt.savefig('analysis_1_binary_msil_test.png', dpi=300)
plt.show()
##################################################################



###################### Analysis 2: binary mixed #####################
print("\n--- Analysis 2: binaire mixed (Non-MSI-H vs MSI-H) ---")
y_2 = y_all.apply(lambda x: 1 if x == 'MSI-H' else 0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_all, y_2, test_size=0.2, random_state=42, stratify=y_2)

X_res_2, y_res_2 = rus.fit_resample(X_train_2, y_train_2)
rs_2 = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=cv_strategy, scoring='f1', n_jobs=-1, random_state=42)
rs_2.fit(X_res_2, y_res_2)
y_pred_2 = rs_2.predict(X_test_2)

###################### Feature importance Analysis 2 ######################
# Get the best tuned random forest model
best_model_2 = rs_2.best_estimator_

# Extract feature importance and attach gene names
feature_importance_2 = pd.DataFrame({
    'gene': X_all.columns,
    'importance': best_model_2.feature_importances_
}).sort_values(by='importance', ascending=False)

# Direction of expression (based on original training split, before undersampling) 
# y_2: 0 = Non-MSI-H, 1 = MSI-H
X_train_2_df = X_train_2.copy()
X_train_2_df['group'] = y_train_2.values

mean_non_msi_h_2 = X_train_2_df[X_train_2_df['group'] == 0].drop(columns='group').mean()
mean_msi_h_2     = X_train_2_df[X_train_2_df['group'] == 1].drop(columns='group').mean()

direction_2 = pd.DataFrame({
    'gene': X_all.columns,
    'mean_non_msi_h': mean_non_msi_h_2.values,
    'mean_msi_h': mean_msi_h_2.values
})

direction_2['higher_in'] = direction_2.apply(
    lambda row: 'MSI-H' if row['mean_msi_h'] > row['mean_non_msi_h'] else 'Non-MSI-H',
    axis=1
)

# Merge feature importance with direction info
feature_importance_2 = feature_importance_2.merge(direction_2, on='gene')

# Save full ranked importance table
feature_importance_2.to_csv('analysis_2_feature_importance_full.csv', index=False)

# Top 10 genes
top10_2 = feature_importance_2.head(10)
print("\nTop 10 important genes - Analysis 2 (Non-MSI-H vs MSI-H):")
print(top10_2)

# Save top 10 genes table
top10_2.to_csv('analysis_2_top10_genes_with_direction.csv', index=False)

# Plot top 10 feature importances
top10_plot_2 = top10_2.sort_values('importance')

plt.figure(figsize=(10, 6))
plt.barh(top10_plot_2['gene'], top10_plot_2['importance'], color='forestgreen')
plt.xlabel('Feature importance')
plt.ylabel('Gene')
plt.title('Top 10 Important Genes - Analysis 2 (Non-MSI-H vs MSI-H)')
plt.tight_layout()
plt.savefig('analysis_2_top10_feature_importance.png', dpi=300)
plt.show()
##########################################################################################

# Visualisation analysis 2
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cm2 = confusion_matrix(y_test_2, y_pred_2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax1, xticklabels=['Non-MSI-H', 'MSI-H'], yticklabels=['Non-MSI-H', 'MSI-H'])
ax1.set_title('Confusion Matrix (Binairy Mixed)')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

report2 = classification_report(y_test_2, y_pred_2, target_names=['Non-MSI-H', 'MSI-H'], output_dict=True)
pd.DataFrame(report2).iloc[:-1, :2].T.plot(kind='bar', ax=ax2, color=metric_color)
ax2.set_title('Classification Metrics')
plt.suptitle('Binary Mixed (MSS+MSI-L VS MSI-H)')
plt.tight_layout()
plt.savefig('analysis_2_binary_mixed.png', dpi=300)
plt.show()
####################################################################


################### Analysis 3: multiclass #################################
print("\n--- Analysis 3: Multiclass (MSS vs MSI-H vs MSI-L) ---")
y_3 = y_all.map({'MSS': 0, 'MSI-H': 1, 'MSI-L': 2})
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_all, y_3, test_size=0.2, random_state=42, stratify=y_3)

rus_multi = RandomUnderSampler(random_state=42, sampling_strategy='not minority')
X_res_3, y_res_3 = rus_multi.fit_resample(X_train_3, y_train_3)
rs_3 = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=cv_strategy, scoring='f1_macro', n_jobs=-1, random_state=42)
rs_3.fit(X_res_3, y_res_3)
y_pred_3 = rs_3.predict(X_test_3)

###################### Feature importance + direction: Analysis 3 ######################
# Get the best tuned random forest model
best_model_3 = rs_3.best_estimator_

# Extract feature importance and attach gene names
feature_importance_3 = pd.DataFrame({
    'gene': X_all.columns,
    'importance': best_model_3.feature_importances_
}).sort_values(by='importance', ascending=False)

# Direction of expression (based on original training split) 
X_train_3_df = X_train_3.copy()
X_train_3_df["group"] = y_train_3.values

# Mean expression per class
group_means_3 = X_train_3_df.groupby("group").mean().T.reset_index()

# Force first column name to be 'gene'
group_means_3.columns = ["gene"] + list(group_means_3.columns[1:])

# Keep only numeric class columns
numeric_cols = [
    col for col in group_means_3.columns
    if col != "gene" and pd.api.types.is_numeric_dtype(group_means_3[col])
]

# Highest / lowest class for each gene
group_means_3["highest_in"] = group_means_3[numeric_cols].astype(float).idxmax(axis=1)
group_means_3["lowest_in"] = group_means_3[numeric_cols].astype(float).idxmin(axis=1)

# Merge feature importance with direction info
feature_importance_3 = feature_importance_3.merge(group_means_3, on="gene")

# Save full ranked importance table
feature_importance_3.to_csv('analysis_3_feature_importance_full.csv', index=False)

# Top 10 genes
top10_3 = feature_importance_3.head(10)
print("\nTop 10 important genes - Analysis 3 (Multiclass):")
print(top10_3)

# Save top 10 genes table
top10_3.to_csv('analysis_3_top10_genes_with_direction.csv', index=False)

# Plot top 10 feature importances
top10_plot_3 = top10_3.sort_values('importance')

plt.figure(figsize=(10, 6))
plt.barh(top10_plot_3['gene'], top10_plot_3['importance'], color='steelblue')
plt.xlabel('Feature importance')
plt.ylabel('Gene')
plt.title('Top 10 Important Genes - Analysis 3 (Multiclass)')
plt.tight_layout()
plt.savefig('analysis_3_top10_feature_importance.png', dpi=300)
plt.show()
##########################################################################################

# Visualisatie Analyse 3
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cm3 = confusion_matrix(y_test_3, y_pred_3)
sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=['MSS', 'MSI-H', 'MSI-L'], yticklabels=['MSS', 'MSI-H', 'MSI-L'])
ax1.set_title('Confusion Matrix (Multiclass)')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

report3 = classification_report(y_test_3, y_pred_3, target_names=['MSS', 'MSI-H', 'MSI-L'], output_dict=True)
pd.DataFrame(report3).iloc[:-1, :3].T.plot(kind='bar', ax=ax2, color=metric_color)
ax2.set_title('Classification Metrics')
plt.suptitle('Multiclass Classification')
plt.tight_layout()
plt.savefig('analysis_3_multiclass.png', dpi=300)
plt.show()
############################################################################
