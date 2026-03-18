import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gseapy as gp
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

filtered_rna_data = pd.read_csv("filtered_rna_data.csv")
filtered_prediction_data = pd.read_csv("filtered_prediction_data.csv").set_index("Unnamed: 0")

rna_matrix = filtered_rna_data.set_index("Gene_Name").T

gsva_results = gp.ssgsea(
    data=rna_matrix.T,
    gene_sets="MSigDB_Hallmark_2020",
    sample_norm_method="rank",
    outdir=None,
    no_plot=True,
    threads=1
)

pathway_scores = gsva_results.res2d
pathway_mat = pathway_scores.pivot(index="Name", columns="Term", values="NES")

common_ids = pathway_mat.index.intersection(filtered_prediction_data.index)
pathway_mat = pathway_mat.loc[common_ids]

#Make numeric
pathway_mat = pathway_mat.apply(pd.to_numeric, errors="coerce")

y_all = filtered_prediction_data.loc[common_ids, "msi_status"]
y_binary = y_all.apply(lambda x: 1 if x == "MSI-H" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    pathway_mat, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Pathway": pathway_mat.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop MSI-H driving pathways:")
print(importance.head(10))

top_pathways = importance["Pathway"].head(10).tolist()

pathway_with_labels = pathway_mat.copy()
pathway_with_labels["msi_status"] = y_all

for pathway in top_pathways:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=pathway_with_labels, x="msi_status", y=pathway)
    sns.stripplot(
        data=pathway_with_labels,
        x="msi_status",
        y=pathway,
        color="black",
        alpha=0.4,
        jitter=True
    )
    plt.title(f"{pathway} activity by MSI status")
    plt.tight_layout()
    plt.show()

    msi_h = pd.to_numeric(
        pathway_with_labels.loc[pathway_with_labels["msi_status"] == "MSI-H", pathway],
        errors="coerce"
    ).dropna()

    non_msi_h = pd.to_numeric(
        pathway_with_labels.loc[pathway_with_labels["msi_status"] != "MSI-H", pathway],
        errors="coerce"
    ).dropna()

    stat, pval = mannwhitneyu(msi_h, non_msi_h, alternative="two-sided")
    print(f"{pathway}: p-value = {pval:.4e}")