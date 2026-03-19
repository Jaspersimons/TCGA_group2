import pandas as pd
import numpy as np
import gseapy as gp
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import os

# ===============================
# Load data
# ===============================

base_path = "/Users/mohammedmoustati/Documents/Master Bioinformatics & Systems biology/Scientific Machine Learning/TCGA_dataset 2"
os.chdir(base_path)

filtered_rna_data = pd.read_csv("filtered_rna_data.csv")
filtered_prediction_data = pd.read_csv("filtered_prediction_data.csv").set_index("Unnamed: 0")

rna_matrix = filtered_rna_data.set_index("Gene_Name").T

# ===============================
# ssGSEA
# ===============================

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

# keep common samples
common_ids = pathway_mat.index.intersection(filtered_prediction_data.index)
pathway_mat = pathway_mat.loc[common_ids]
pathway_mat = pathway_mat.apply(pd.to_numeric, errors="coerce")

# labels
y_all = filtered_prediction_data.loc[common_ids, "msi_status"]
y_binary = y_all.apply(lambda x: 1 if x == "MSI-H" else 0)

# ===============================
# Random Forest → get top pathways
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    pathway_mat, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Pathway": pathway_mat.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

top_pathways = importance["Pathway"].head(10).tolist()

print("\nTop pathways:")
print(top_pathways)

# save pathway importance
importance.to_csv("pathway_importance_results.csv", index=False)

# save top pathways only
pd.DataFrame({"Top_Pathways": top_pathways}).to_csv("top_pathways.csv", index=False)

# ===============================
# Gene-level analysis
# ===============================

gene_sets = gp.get_library(name="MSigDB_Hallmark_2020")

results = []

for pathway in top_pathways:
    if pathway not in gene_sets:
        continue

    genes = gene_sets[pathway]
    genes = [g for g in genes if g in rna_matrix.columns]

    for gene in genes:
        msi_h = rna_matrix.loc[y_all == "MSI-H", gene]
        non_msi_h = rna_matrix.loc[y_all != "MSI-H", gene]

        msi_h = pd.to_numeric(msi_h, errors="coerce").dropna()
        non_msi_h = pd.to_numeric(non_msi_h, errors="coerce").dropna()

        if len(msi_h) < 5 or len(non_msi_h) < 5:
            continue

        stat, pval = mannwhitneyu(msi_h, non_msi_h)
        effect = msi_h.mean() - non_msi_h.mean()

        results.append({
            "Pathway": pathway,
            "Gene": gene,
            "Effect": effect,
            "p_value": pval
        })

# results table
gene_df = pd.DataFrame(results)
gene_df = gene_df.sort_values(by="Effect", ascending=False)

print("\nTop genes driving pathways:")
print(gene_df.head(20))

# save full gene results
gene_df.to_csv("gene_level_pathway_results.csv", index=False)

# save top 20 genes
gene_df.head(20).to_csv("top20_genes_driving_pathways.csv", index=False)

# ===============================
# Pathway–Gene network
# ===============================

# keep top 5 genes per pathway
network_df = (
    gene_df.sort_values(["Pathway", "Effect"], ascending=[True, False])
    .groupby("Pathway")
    .head(5)
)

# save network table
network_df.to_csv("pathway_gene_network_table.csv", index=False)

# build graph
G = nx.Graph()

for pathway in network_df["Pathway"].unique():
    G.add_node(pathway, node_type="pathway")

for _, row in network_df.iterrows():
    pathway = row["Pathway"]
    gene = row["Gene"]
    G.add_node(gene, node_type="gene")
    G.add_edge(pathway, gene)

pos = nx.spring_layout(G, k=0.8, seed=42)

pathway_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "pathway"]
gene_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "gene"]

plt.figure(figsize=(16, 12))

nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=pathway_nodes,
    node_size=3000,
    node_color="lightcoral"
)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=gene_nodes,
    node_size=1200,
    node_color="lightblue"
)
nx.draw_networkx_labels(G, pos, font_size=9)

plt.title("Pathway–Gene Network of MSI-H Driving Pathways")
plt.axis("off")
plt.tight_layout()

# save figure
plt.savefig("pathway_gene_network.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nFiles saved in:")
print(base_path)
print("""
Saved files:
- pathway_importance_results.csv
- top_pathways.csv
- gene_level_pathway_results.csv
- top20_genes_driving_pathways.csv
- pathway_gene_network_table.csv
- pathway_gene_network.png
""")