# ==========================================
# Linear SVM Feature Importance (Python)
# Multiclass + Binary + Pathway Enrichment
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp

np.random.seed(42)

# ---------------------------
# 1. Load data
# ---------------------------
rna_data = pd.read_csv("filtered_dna_data.csv", index_col=0)
prediction_data = pd.read_csv("filtered_prediction_data.csv", index_col=0)

# Match IDs
common_ids = list(set(rna_data.columns).intersection(prediction_data.index))

X_all = rna_data[common_ids].T
y_all = prediction_data.loc[common_ids, "msi_status"]

# Remove Indeterminate
mask = y_all != "Indeterminate"
X_all = X_all[mask]
y_all = y_all[mask]

# ---------------------------
# 2. MULTICLASS SVM
# ---------------------------
print("\n=== MULTICLASS SVM ===")

classes = ["MSI-H", "MSI-L", "MSS"]
y_multi = pd.Categorical(y_all, categories=classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_multi, test_size=0.2, stratify=y_multi, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
svm_multi = LinearSVC()
svm_multi.fit(X_train_scaled, y_train)

# Predict
y_pred = svm_multi.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Multiclass)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# ---------------------------
# Feature importance
# ---------------------------
coef = svm_multi.coef_
importance = np.max(np.abs(coef), axis=0)

feature_importance = pd.DataFrame({
    "gene": X_all.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

# Mean expression per class
train_df = X_train.copy()
train_df["group"] = y_train.values

means = train_df.groupby("group").mean().T
means["higher_in"] = means.idxmax(axis=1)

feature_importance = feature_importance.merge(
    means["higher_in"], left_on="gene", right_index=True
)

# Top 10 plot
top10 = feature_importance.head(10)

plt.figure()
sns.barplot(data=top10, x="importance", y="gene", hue="higher_in")
plt.title("Top 10 Genes (Multiclass SVM)")
plt.show()

# ---------------------------
# Pathway enrichment (MSI-H)
# ---------------------------
top_msi = feature_importance[
    feature_importance["higher_in"] == "MSI-H"
].head(50)["gene"].tolist()

enr = gp.enrichr(
    gene_list=top_msi,
    gene_sets="MSigDB_Hallmark_2020",
    organism="Human",
    outdir=None
)

print(enr.results.head(10))


# ---------------------------
# 3. BINARY SVM
# ---------------------------
print("\n=== BINARY SVM ===")

y_bin = np.where(y_all == "MSI-H", "MSI-H", "Non-MSI-H")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_bin, test_size=0.2, stratify=y_bin, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
svm_bin = LinearSVC()
svm_bin.fit(X_train_scaled, y_train)

# Predict
y_pred = svm_bin.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["MSI-H", "Non-MSI-H"])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["MSI-H", "Non-MSI-H"],
            yticklabels=["MSI-H", "Non-MSI-H"])
plt.title("Confusion Matrix (Binary)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

print(classification_report(y_test, y_pred))

# ---------------------------
# Feature importance
# ---------------------------
coef = svm_bin.coef_[0]
importance = np.abs(coef)

feature_importance_bin = pd.DataFrame({
    "gene": X_all.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

# Direction
mean_msi = X_train[y_train == "MSI-H"].mean()
mean_non = X_train[y_train == "Non-MSI-H"].mean()

direction = pd.DataFrame({
    "gene": X_all.columns,
    "higher_in": np.where(mean_msi > mean_non, "MSI-H", "Non-MSI-H")
})

feature_importance_bin = feature_importance_bin.merge(direction, on="gene")

# Top 10 plot
top10 = feature_importance_bin.head(10)

plt.figure()
sns.barplot(data=top10, x="importance", y="gene", hue="higher_in")
plt.title("Top 10 Genes (Binary SVM)")
plt.show()

# ---------------------------
# Pathway enrichment
# ---------------------------
top_msi = feature_importance_bin[
    feature_importance_bin["higher_in"] == "MSI-H"
].head(50)["gene"].tolist()

enr_bin = gp.enrichr(
    gene_list=top_msi,
    gene_sets="MSigDB_Hallmark_2020",
    organism="Human",
    outdir=None
)

print(enr_bin.results.head(10))
