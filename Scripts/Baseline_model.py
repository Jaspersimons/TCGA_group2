import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


############ Load the preprocessed data ##################
rna_data = pd.read_csv('filtered_rna_data.csv')

prediction_data = pd.read_csv('filtered_prediction_data.csv')
#####################################################


################# Align by sample ID #########################
# Ensure both datasets are aligned by sample IDs (matching the valid samples)
# Assuming that the first column in the RNA data has the gene names and the first column in the prediction data has sample IDs
valid_samples = prediction_data['Unnamed: 0']

# Align RNA data and prediction data by the sample IDs (columns for RNA data, rows for prediction data)
rna_data_filtered = rna_data.loc[:, rna_data.columns.isin(valid_samples)]  # Keep only valid samples in RNA data

prediction_data_filtered = prediction_data[prediction_data['Unnamed: 0'].isin(valid_samples)]  # Filter valid samples in prediction data

# Ensure that both datasets now have matching samples
assert rna_data_filtered.shape[1] == prediction_data_filtered.shape[0], "Mismatch in number of samples between RNA data and prediction data"

############################################################



######################## Prepare features and target variable ################################
X = rna_data_filtered.T  # Transpose so that samples are rows and genes are columns
y = prediction_data_filtered['msi_status']  # Set target variable: MSI status
######################################################################################


################# Train/test split ##################
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
###################################################


############################# Standardize data (for logisitic regression) ############################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
###################################################################################


################################ Logistic regression #################################
# Define the Baseline Model: Logistic Regression with class weight adjustment
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_test_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
print("Logistic Regression - Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Logistic Regression - Test Accuracy Score:")
print(accuracy_score(y_test, y_test_pred))

####################################################################################



################### Random forest ###############################
# Random Forest as another baseline model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_clf.fit(X_train_scaled, y_train)



# Make predictions with Random Forest
y_pred_rf = rf_clf.predict(X_test_scaled)

print("\nRandom Forest - Test Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Random Forest - Test Accuracy Score:")
print(accuracy_score(y_test, y_pred_rf))
#########################################################



############### Confusion matrix #############################
# Display confusion matrix
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_test_pred))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
######################################################################



################## Plots: confusion matrix #####################
class_names = ['MSI-H', 'MSI-L', 'MSS']

# Compute confusion matrices
cm_log = confusion_matrix(y_test, y_test_pred)
cm_rf = confusion_matrix(y_test, y_pred_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices_raw.png", dpi=300, bbox_inches='tight')
plt.show()
#########################################################



#################### plots: per class precision, recall, f1 bar #######################

# Get classification reports as dicts
report_log = classification_report(y_test, y_test_pred, 
                                   target_names=class_names, 
                                   output_dict=True)

report_rf = classification_report(y_test, y_pred_rf, 
                                  target_names=class_names, 
                                  output_dict=True)

# Convert to DataFrame
df_log = pd.DataFrame(report_log).T.iloc[:3]
df_rf = pd.DataFrame(report_rf).T.iloc[:3]

# Add model label
df_log['Model'] = 'Logistic Regression'
df_rf['Model'] = 'Random Forest'

df_all = pd.concat([df_log, df_rf])
df_all.reset_index(inplace=True)
df_all.rename(columns={'index': 'Class'}, inplace=True)



metrics = ['precision', 'recall', 'f1-score']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(metrics):
    sns.barplot(data=df_all, x='Class', y=metric, hue='Model', ax=axes[i])
    axes[i].set_title(metric.capitalize())
    axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("model_performance_metrics.png", dpi=300, bbox_inches='tight')
plt.show()

#################################################################################



######################## plots: normalized confusion matrix ################

cm_log_norm = cm_log.astype('float') / cm_log.sum(axis=1)[:, np.newaxis]
cm_rf_norm = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_log_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0])
axes[0].set_title("Logistic Regression (Normalized)")

sns.heatmap(cm_rf_norm, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1])
axes[1].set_title("Random Forest (Normalized)")

plt.tight_layout()
plt.savefig("confusion_matrices_normalized.png", dpi=300, bbox_inches='tight')
plt.show()
###########################################################################





