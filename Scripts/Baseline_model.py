import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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