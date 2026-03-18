import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

############## Configure colors ###################
box_color = '#7fb3d5'
###################################################

################ Load data ##########################
rna_data = pd.read_csv('filtered_rna_data.csv', index_col=0)
prediction_data = pd.read_csv('filtered_prediction_data.csv').set_index('Unnamed: 0')

common_ids = rna_data.columns.intersection(prediction_data.index)
# Assuming X_all currently contains genes. If you have generated pathway scores 
# (like Hallmark scores), load that dataframe here instead of rna_data.
X_all = rna_data[common_ids].T 
y_all = prediction_data.loc[common_ids, 'msi_status']
######################################################

################ Setup Target ########################
# Grouping MSI-L and MSS as 0 (Non-MSI-H) and MSI-H as 1
y_bin = y_all.apply(lambda x: 1 if x == 'MSI-H' else 0)
######################################################

################ Robustness Analysis Loop ############
print("\n--- Running Feature Stability & Robustness Analysis ---")
n_iterations = 50
sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=0.2, random_state=42)

# Using L1 Logistic Regression because it forces irrelevant features to 0 
# and provides positive/negative coefficients to check directional stability.
model = LogisticRegression(
    penalty='elasticnet', 
    l1_ratio=1.0, 
    solver='saga', 
    max_iter=1000, 
    class_weight='balanced', 
    random_state=42
)
# To store the coefficients of every feature across all splits
all_coefs = []

iteration = 1
for train_index, test_index in sss.split(X_all, y_bin):
    if iteration % 10 == 0:
        print(f"Processing split {iteration}/{n_iterations}...")
        
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_bin.iloc[train_index], y_bin.iloc[test_index]
    
    # Scaling is mandatory for L1 regularisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, y_train)
    
    # Store coefficients
    all_coefs.append(model.coef_[0])
    iteration += 1

######################################################

################ Calculate Stability Metrics #########
# Convert list of arrays to DataFrame (Rows = Splits, Columns = Features)
coef_df = pd.DataFrame(all_coefs, columns=X_all.columns)

stability_results = []

for feature in coef_df.columns:
    feature_coefs = coef_df[feature]
    
    # 1. Selection Frequency (How often is the coefficient NOT zero?)
    non_zero_count = (feature_coefs != 0).sum()
    freq = non_zero_count / n_iterations
    
    # 2. Mean Coefficient (magnitude of impact)
    mean_coef = feature_coefs.mean()
    
    # 3. Directional Stability (When selected, does it stay positive or negative?)
    if non_zero_count > 0:
        positive_count = (feature_coefs > 0).sum()
        negative_count = (feature_coefs < 0).sum()
        # % of time it takes its dominant direction
        directional_consistency = max(positive_count, negative_count) / non_zero_count 
    else:
        directional_consistency = 0
        
    stability_results.append({
        'Feature': feature,
        'Selection_Frequency': freq,
        'Mean_Coefficient': mean_coef,
        'Directional_Consistency': directional_consistency
    })

results_df = pd.DataFrame(stability_results)

# Filter for the most robust features:
# Selected in at least 50% of splits AND 100% directionally consistent when selected
robust_features = results_df[
    (results_df['Selection_Frequency'] >= 0.50) & 
    (results_df['Directional_Consistency'] >= 0.95)
].sort_values(by='Selection_Frequency', ascending=False)

print("\nTop 15 Most Robust Features:")
print(robust_features.head(15).to_string(index=False))
######################################################

################ Visualization #######################
if not robust_features.empty:
    top_n = min(15, len(robust_features))
    top_feature_names = robust_features.head(top_n)['Feature'].tolist()
    
    # Extract just the data for the top features across all splits
    plot_data = coef_df[top_feature_names].melt(var_name='Feature', value_name='Coefficient')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=plot_data, x='Coefficient', y='Feature', color=box_color, showfliers=False)
    sns.stripplot(data=plot_data, x='Coefficient', y='Feature', color='black', alpha=0.3, size=3)
    
    # Add a vertical line at 0 to clearly show if any features cross the zero bound (changing direction)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    plt.title(f'Coefficient Stability Across {n_iterations} Resampling Splits\n(Features reliably predicting MSI-H)')
    plt.xlabel('Logistic Regression Coefficient (L1)')
    plt.ylabel('Feature / Pathway')
    plt.tight_layout()
    plt.savefig('feature_stability_analysis.png', dpi=300)
    plt.show()
else:
    print("\nNo features met the strict robustness threshold (>=50% frequency & >=95% directional consistency).")
######################################################