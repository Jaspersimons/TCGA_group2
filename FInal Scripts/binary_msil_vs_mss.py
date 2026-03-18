import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


def main():
    # --- 1) Check required files exist ---
    required_files = ["filtered_rna_data.csv", "filtered_prediction_data.csv"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s): {missing}\n"
            "Put this script in the same folder as the filtered CSV files, "
            "or run it from that folder."
        )

    # --- 2) Load data ---
    rna_data = pd.read_csv("filtered_rna_data.csv")
    pred_data = pd.read_csv("filtered_prediction_data.csv")

    # --- 3) Build X (samples x genes) ---
    # Assumption: first column = gene names, remaining columns = samples
    X = rna_data.iloc[:, 1:].T

    # --- 4) Build y (MSI labels) ---
    if "msi_status" in pred_data.columns:
        y = pred_data["msi_status"]
    elif "MSI" in pred_data.columns:
        y = pred_data["MSI"]
    else:
        raise ValueError(
            f"Could not find MSI label column. Available columns: {list(pred_data.columns)}"
        )

    # --- 5) Align X and y by sample IDs ---
    # We assume sample IDs are the column names of RNA data (after transpose),
    # and are either the index of pred_data or in a column inside pred_data.
    X.index = X.index.astype(str)

    # Case A: pred_data already uses sample IDs as its index
    y_series = y.copy()
    y_series.index = pred_data.index.astype(str)

    # If pred_data has an explicit sample ID column, use it instead (common names)
    possible_id_cols = ["Unnamed: 0", "sample_id", "Sample_ID", "sample", "Sample", "barcode", "Barcode"]
    id_col = None
    for c in possible_id_cols:
        if c in pred_data.columns:
            id_col = c
            break

    if id_col is not None:
        y_series.index = pred_data[id_col].astype(str)
    else:
        raise ValueError(
            "Could not find a sample ID column in prediction data. "
            "Tried: " + ", ".join(possible_id_cols)
        )

    # Now align
    common = X.index.intersection(y_series.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping sample IDs between RNA data and prediction data.\n"
            "This usually means the prediction file does not contain sample IDs "
            "in its index or in a recognizable ID column.\n"
            f"RNA sample ID examples: {list(X.index[:5])}\n"
            f"Prediction index examples: {list(y_series.index[:5])}\n"
            f"Prediction columns: {list(pred_data.columns)}"
        )

    X = X.loc[common]
    y_series = y_series.loc[common]

    # --- 6) Keep only MSS and MSI-L ---
    mask = y_series.isin(["MSS", "MSI-L"])
    X_bin = X.loc[mask]
    y_bin = y_series.loc[mask]

    print("=== Binary task: MSI-L vs MSS ===")
    print("Binary dataset size (samples, genes):", X_bin.shape)
    print("\nClass counts (all):")
    print(y_bin.value_counts())

    if y_bin.nunique() < 2:
        raise ValueError(
            "After filtering to MSS and MSI-L, only one class remains.\n"
            "Check that your labels contain exactly 'MSS' and 'MSI-L'."
        )

    # --- 7) Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    print("\nClass counts (test):")
    print(y_test.value_counts())

    # --- 8) Train model ---
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 9) Metrics ---
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    labels = ["MSI-L", "MSS"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion matrix (rows=true, cols=pred) [MSI-L, MSS]:")
    print(cm)

    # --- 10) Normalized confusion matrix + save figure ---
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Binary MSI-L vs MSS (Normalized Confusion Matrix)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    outname = "binary_msil_vs_mss_confusion_normalized.png"
    plt.savefig(outname, dpi=200)
    print(f"\nSaved figure: {outname}")

    plt.show()


if __name__ == "__main__":
    main()