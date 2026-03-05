import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import randint
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

############## Configure colors for fun ###################
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