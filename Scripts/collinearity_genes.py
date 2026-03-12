import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

import gseapy as gp

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

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)


# ===============================
# Feature importance
# ===============================

importance = pd.DataFrame({
    "Pathway": pathway_mat.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values("Importance", ascending=False)

print("\nTop MSI-H driving pathways:")
print(importance.head(15))


# ===============================
# Plot
# ===============================

plt.figure(figsize=(8,6))
sns.barplot(data=importance.head(15), y="Pathway", x="Importance")
plt.title("Top MSI-H Driving Pathways")
plt.tight_layout()
plt.show()