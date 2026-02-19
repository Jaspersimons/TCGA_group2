import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


#input file
df = pd.read_csv("prediction_file_crc.csv")
# =============


#remove indeterminate genes for chi square
df = df[df["msi_status"] != "Indeterminate"]


genes = ["TP53", "KRAS", "BRAF", "APC", "TTN"]

#create empty list
results = []


#contingency and chi square for every gene in list genes
for gene in genes:

    #selects msi status column; drops rows with NAN values
    sub = df[["msi_status", gene]].dropna()

    #convert WT/SNV to 0/1
    sub[gene] = sub[gene].map({"WT":0, "SNV":1})

    #make contingency table (shows distribution of WT/SNV per MSI status per gene)
    contingency = pd.crosstab(sub["msi_status"], sub[gene])

    # Save table for fun per gene
    contingency.to_csv(gene + "_contingency_table.csv")

    #debug line print
    ####print(contingency)


    #perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    results.append([gene, chi2, p_value])

    # stacked bar plot with contingency
    proportions = contingency.div(contingency.sum(axis=1), axis=0)

    proportions.plot(kind="bar", stacked=True)
    plt.title(gene + " Mutation Across MSI Classes")
    plt.ylabel("Proportion")
    plt.xlabel("MSI Status")
    plt.legend(["Wildtype", "Mutated"])
    plt.tight_layout()
    plt.savefig(gene + "_stacked_barplot.png", dpi=300)
    plt.close()


#save summary results of chi square
results_df = pd.DataFrame(results, columns=["Gene", "Chi2_statistic", "Chi2_pvalue"])

#save to csv
results_df.to_csv("results_chi2_mutation_enrichment.csv", index=False)

#figure for Chi-square results
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

#color bars based on p-value significance (p < 0.05 is significant)
colors = ['green' if p < 0.05 else 'gray' for p in results_df['Chi2_pvalue']]
bars = ax.barh(results_df["Gene"], results_df["Chi2_statistic"], color = colors, align="center")

#scatter plot for p-values to overlay
ax.scatter(results_df['Chi2_pvalue'], results_df['Gene'], color='red', label="p-value", zorder=5)

#annotate p-values next to the points
for i, p_val in enumerate(results_df['Chi2_pvalue']):
    ax.text(results_df['Chi2_statistic'][i] + 5, i, f'{p_val:.2e}', va='center', color='red')

#add labels and title
plt.xlabel('Chi2 Statistic and p-value')
plt.ylabel('Genes')
plt.title('Chi2 Statistic and p-values for Each Gene')
#add grid lines and formatting
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

#print figure
plt.savefig("chi_square_results_figure.png", dpi=300)
plt.show()