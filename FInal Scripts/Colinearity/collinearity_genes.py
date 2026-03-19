import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import gseapy as gp

# ===============================
# Load data
# ===============================

os.chdir("/Users/mohammedmoustati/Documents/Master Bioinformatics & Systems biology/Scientific Machine Learning/TCGA_dataset 2")

filtered_rna_data = pd.read_csv("filtered_rna_data.csv")
filtered_prediction_data = pd.read_csv("filtered_prediction_data.csv").set_index("Unnamed: 0")

# genes x samples → samples x genes
rna_matrix = filtered_rna_data.set_index("Gene_Name").T


# ===============================
# ssGSEA pathway scores
# ===============================

gsva_results = gp.ssgsea(
    data=rna_matrix.T,
    gene_sets="MSigDB_Hallmark_2020",
    sample_norm_method="rank",
    outdir=None
)

pathway_scores = gsva_results.res2d

# long → matrix
pathway_mat = pathway_scores.pivot(index="Name", columns="Term", values="NES")


# ===============================
# Prepare labels
# ===============================

y_all = filtered_prediction_data.loc[pathway_mat.index, "msi_status"]

# binary classification: MSI-H vs others
y_binary = y_all.apply(lambda x: 1 if x == "MSI-H" else 0)


# ===============================
# Train Random Forest
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    pathway_mat, y_binary,
    test_size=0.2,
    stratify=y_binary,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
rf.fit(X_train, y_train)


# ===============================
# Predictions and performance
# ===============================

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Non-MSI-H", "MSI-H"]
)

print(f"\nAccuracy: {acc:.4f}")
print("\nClassification report:")
print(report)


# ===============================
# Feature importance + error bars
# ===============================

# gemiddelde importance over forest
importances = rf.feature_importances_

# standaarddeviatie van importance over alle trees
std = np.std(
    [tree.feature_importances_ for tree in rf.estimators_],
    axis=0
)

importance = pd.DataFrame({
    "Pathway": pathway_mat.columns,
    "Importance": importances,
    "Std": std
}).sort_values("Importance", ascending=False)

print("\nTop MSI-H driving pathways:")
print(importance.head(15))


# ===============================
# Save results
# ===============================

output_dir = "results_msi_rf"
os.makedirs(output_dir, exist_ok=True)

# Save full importance table
importance.to_csv(os.path.join(output_dir, "all_pathway_importance.csv"), index=False)

# Save top 15 pathways
top15 = importance.head(15)
top15.to_csv(os.path.join(output_dir, "top15_pathways.csv"), index=False)

# Save model
joblib.dump(rf, os.path.join(output_dir, "random_forest_model.pkl"))

# Save performance metrics
with open(os.path.join(output_dir, "model_performance.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)

# Save confusion matrix values
cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-MSI-H", "Actual MSI-H"],
    columns=["Predicted Non-MSI-H", "Predicted MSI-H"]
)
cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))


# ===============================
# Plot 1: standard feature importance
# ===============================

plt.figure(figsize=(8, 6))
sns.barplot(data=top15, y="Pathway", x="Importance")
plt.title("Top MSI-H Driving Pathways")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top15_pathways.png"), dpi=300)
plt.close()


# ===============================
# Plot 2: feature importance with error bars
# ===============================

plt.figure(figsize=(10, 7))
plt.barh(
    top15["Pathway"][::-1],
    top15["Importance"][::-1],
    xerr=top15["Std"][::-1],
    capsize=4
)
plt.xlabel("Feature Importance")
plt.ylabel("Pathway")
plt.title("Top 15 MSI-H Driving Pathways with Error Bars")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top15_pathways_errorbars.png"), dpi=300)
plt.close()


# ===============================
# Plot 3: confusion matrix
# ===============================

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-MSI-H", "MSI-H"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.close()


print(f"\nResults saved in folder: {output_dir}")
