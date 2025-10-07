import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from xgboost import plot_importance
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np, pandas as pd
import uproot
import pyhf

outputfolder = 'Run5'

def PlotROC(y_true_train, y_score_train, y_true_test, y_score_test):
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_score_train)
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.3f})", linestyle='--', color='springgreen')
    plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.3f})", linestyle='-', color='tomato')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Receiver Operating Characteristic) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def PlotScoreDistributions(y_true_train, y_score_train, y_true_test, y_score_test):
    plt.figure(figsize=(10,6))
    sns.histplot(y_score_train[y_true_train == 0], label="Train: Class 0", color="mediumspringgreen", stat="density", bins=50, alpha=0.5)
    sns.histplot(y_score_train[y_true_train == 1], label="Train: Class 1", color="tomato", stat="density", bins=50, alpha=0.5)
    sns.histplot(y_score_test[y_true_test == 0], label="Test: Class 0", color="mediumspringgreen", stat="density", bins=50, alpha=0.2, linestyle="--")
    sns.histplot(y_score_test[y_true_test == 1], label="Test: Class 1", color="tomato", stat="density", bins=50, alpha=0.2, linestyle="--")
    
    plt.xlabel("Model Score")
    plt.ylabel("Density")
    plt.title("Model Output Score Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def PlotFeatureImportanceXGB(model, feature_names=None):
    plot_importance(model, importance_type='gain', max_num_features=30, show_values=False)
    plt.title("Top Feature Importances (Gain)")
    plt.tight_layout()
    plt.show()

def PlotFeatureImportanceSci(model, feature_names=None):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names)
    sorted_importances = feature_importances.sort_values()
    plt.figure(figsize=(10, 8))
    sorted_importances.plot(kind='barh', color='palevioletred')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def PlotROCpdf(y_true_train, y_score_train, y_true_test, y_score_test, ax):
    """Draws the ROC curve on a given matplotlib axis."""
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_score_train)
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    ax.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.3f})", linestyle='--', color='springgreen')
    ax.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.3f})", linestyle='-', color='tomato')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (Receiver Operating Characteristic) Curve")
    ax.legend(loc="lower right")
    ax.grid(True)

def PlotScoreDistributionspdf(y_true_train, y_score_train, y_true_test, y_score_test, ax):
    """Draws score distributions on a given matplotlib axis."""
    sns.histplot(y_score_train[y_true_train == 0], label="Train: Class 0", color="mediumspringgreen", stat="density", bins=50, alpha=0.5, ax=ax)
    sns.histplot(y_score_train[y_true_train == 1], label="Train: Class 1", color="tomato", stat="density", bins=50, alpha=0.5, ax=ax)
    sns.histplot(y_score_test[y_true_test == 0], label="Test: Class 0", color="mediumspringgreen", stat="density", bins=50, alpha=0.2, linestyle="--", ax=ax)
    sns.histplot(y_score_test[y_true_test == 1], label="Test: Class 1", color="tomato", stat="density", bins=50, alpha=0.2, linestyle="--", ax=ax)
    ax.set_xlabel("Model Score")
    ax.set_ylabel("Density")
    ax.set_title("Model Output Score Distributions")
    ax.legend()
    ax.grid(True)

def PlotFeatureImportanceScipdf(model, feature_names, ax):
    """Draws feature importances on a given matplotlib axis."""
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names)
    sorted_importances = feature_importances.sort_values()
    sorted_importances.plot(kind='barh', color='palevioletred', ax=ax)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Importance')

def pdfplot(title, X_train, X_test, y_train, y_test, model, feature_names=None):
    OutputDirectory = Path(__file__).parent.parent / 'OutputData' / outputfolder
    OutputDirectory.mkdir(parents=True, exist_ok=True)
    title = OutputDirectory / title
    y_score_train = model.predict_proba(X_train)[:, 1]
    y_score_test = model.predict_proba(X_test)[:, 1]

    with PdfPages(title) as pdf:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        PlotROCpdf(y_train, y_score_train, y_test, y_score_test, ax=ax1)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        PlotScoreDistributionspdf(y_train, y_score_train, y_test, y_score_test, ax=ax2)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

        if feature_names is not None:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            PlotFeatureImportanceScipdf(model, feature_names, ax=ax3)
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)

    print(f"PDF with all plots has been saved as {title}")

def Punzi(y_test, y_score, n_sigma=3):
    """
    Calculates the optimal BDT cut using the Punzi significance method.

    Args:
        y_true (array-like): True binary labels.
        y_score (array-like): Target scores, can be probability estimates of the positive class.
        n_sigma (float): The desired significance level in units of sigma.

    Returns:
        None: Prints the optimal cut and displays a plot.
    """

    OutputDirectory = Path(__file__).parent.parent / 'OutputData' / outputfolder
    OutputDirectory.mkdir(parents=True, exist_ok=True)
    title = OutputDirectory / "Punzi_Significance.pdf"

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    s_total = np.sum(y_test == 1)
    b_total = np.sum(y_test == 0)
    print(f"Total signal events: {s_total}, Total background events: {b_total}")
    signal_efficiency = tpr
    background_events = fpr * b_total

    ######################################################
    # Calculate Punzi significance, avoiding division by #
    # zero. Adding a small epsilon to the denominator to #
    # prevent issues with log(0). This is a simplified   #
    # version of the Punzi significance calculation. The #
    # significance formula: S / sqrt(S + b + 1e-9) where #
    # S is the signal efficiency and b is the background #
    # events. Use n_sigma to scale the background events #
    # to the desired significance level.                 #
    ######################################################
    punzi_significance = signal_efficiency / (n_sigma/2 + np.sqrt(background_events))

    # Find the threshold that maximizes the significance
    optimal_idx = np.argmax(punzi_significance)
    optimal_threshold = thresholds[optimal_idx]
    max_significance = punzi_significance[optimal_idx]

    print("\n--- Punzi Optimization ---")
    print(f"Optimal BDT score cut: {optimal_threshold:.4f}")
    print(f"Maximum Punzi Significance: {max_significance:.4f}")
    
    plt.figure(figsize=(10, 6))
    # We plot against thresholds, but need to remove the first value to match array lengths
    plt.plot(thresholds[1:], punzi_significance[1:], label='Punzi Significance', color='darkcyan')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Cut = {optimal_threshold:.3f}')
    plt.xlabel('BDT Score Threshold')
    plt.ylabel('Punzi Significance')
    plt.title(f'Punzi Significance vs. BDT Cut ($n_\\sigma$ = {n_sigma})')
    plt.legend()
    plt.grid(True)
    plt.savefig(title)
    plt.show()

def SaveBDTResults(model, features, output_filename, output_filename_ref):
    """
    Applies a trained BDT model to datasets, adds the BDT score and a target label
    as new branches, and saves the result as a new ROOT TTree.

    Args:
        model: The trained scikit-learn/XGBoost model.
        features (list): The list of feature names used for training.
        output_filename (str): The name of the output ROOT file.
    """
    InputDirectory = Path(__file__).parent.parent / 'InputData'
    OutputDirectory = Path(__file__).parent.parent / 'OutputData' / outputfolder
    OutputDirectory.mkdir(parents=True, exist_ok=True)

    with uproot.open(InputDirectory / "Lc2pemu_MC.root") as f:
        df_sig = f["DecayTree"].arrays(library="pd")
    with uproot.open(InputDirectory / "Lc2pemu_DATA_osign_noBrem.root") as f:
        df_bkg = f["DecayTree"].arrays(library="pd")
    with uproot.open(InputDirectory / "Lc2pphimumu_MC.root") as f:
        df_ref_sig = f["DecayTree"].arrays(library="pd")
    with uproot.open(InputDirectory / "Lc2pmumu_DATA.root") as f:
        df_ref_bkg = f["DecayTree"].arrays(library="pd")

    # Add the target column to each dataframe BEFORE combining them
    df_sig['target'] = 1  # Add target=1 for all signal events
    df_bkg['target'] = 0  # Add target=0 for all background events
    df_ref_sig['target'] = 1  # Add target=1 for all reference signal events
    df_ref_bkg['target'] = 0  # Add target=0 for all reference background events

    # Combine data
    df_full = pd.concat([df_sig, df_bkg], ignore_index=True)
    df_ref_full = pd.concat([df_ref_sig, df_ref_bkg], ignore_index=True)
    
    print("Converting mixed-type columns for ROOT compatibility...")
    for col in df_full.columns:
        if df_full[col].dtype == 'object':
            try:
                df_full[col] = pd.to_numeric(df_full[col])
            except (ValueError, TypeError):
                continue

        if df_full[col].dtype == 'bool':
            df_full[col] = df_full[col].astype(int)

    for col in df_ref_full.columns:
        if df_ref_full[col].dtype == 'object':
            try:
                df_ref_full[col] = pd.to_numeric(df_ref_full[col])
            except (ValueError, TypeError):
                continue

        if df_ref_full[col].dtype == 'bool':
            df_ref_full[col] = df_ref_full[col].astype(int)

    X_full = df_full[features]
    X_ref_full = df_ref_full[features]
    bdt_scores = model.predict_proba(X_full)[:, 1]
    bdt_ref_scores = model.predict_proba(X_ref_full)[:, 1]

    df_full['bdt_score'] = bdt_scores
    df_ref_full['bdt_score'] = bdt_ref_scores

    # This boolean conversion loop is redundant if the one above is comprehensive
    for col in df_full.select_dtypes(include='bool').columns:
        df_full[col] = df_full[col].astype(int)
    for col in df_ref_full.select_dtypes(include='bool').columns:
        df_ref_full[col] = df_ref_full[col].astype(int)

    with uproot.recreate(OutputDirectory / output_filename) as f:
        f["DecayTree"] = df_full
    with uproot.recreate(OutputDirectory / output_filename_ref) as f:
        f["DecayTreeRef"] = df_ref_full

    print(f"Successfully saved data with BDT scores and target labels to '{output_filename}' and '{output_filename_ref}'")

def SaveBDTResultsSmall(model, selected_features, output_filename="outputBDT.root"):
    """
    Applies a trained BDT model, saving only the selected features plus the BDT score
    to a new ROOT TTree.

    Args:
        model: The trained BDT model (which should be trained on selected_features).
        selected_features (list): The list of feature names that the BDT selected.
        output_filename (str): The name of the output ROOT file.
    """
    InputDirectory = Path(__file__).parent.parent / 'InputData'
    OutputDirectory = Path(__file__).parent.parent / 'OutputData' / outputfolder
    OutputDirectory.mkdir(parents=True, exist_ok=True)

    # --- CHANGED: Load ONLY the selected features from the input files ---
    # This is much more memory-efficient.
    print(f"Loading {len(selected_features)} selected features from input files...")
    with uproot.open(InputDirectory / "Lc2pemu_MC.root") as f:
        df_sig = f["DecayTree"].arrays(selected_features, library="pd")
    with uproot.open(InputDirectory / "Lc2pemu_DATA_osign_noBrem.root") as f:
        df_bkg = f["DecayTree"].arrays(selected_features, library="pd")

    # Combine the datasets
    df_to_process = pd.concat([df_sig, df_bkg], ignore_index=True)

    # The dataframe `df_to_process` now contains exactly the columns the model expects.
    # We can pass it directly to predict_proba.
    print("Applying BDT model to calculate scores...")
    bdt_scores = model.predict_proba(df_to_process)[:, 1]
    
    # Add the new BDT score as a column
    df_to_process['bdt_score'] = bdt_scores

    # Clean boolean types for ROOT compatibility, if any exist
    for col in df_to_process.select_dtypes(include='bool').columns:
        df_to_process[col] = df_to_process[col].astype(int)

    # The dataframe now contains ONLY the selected features and the bdt_score.
    print(f"Saving {len(df_to_process.columns)} branches to output file...")
    with uproot.recreate(OutputDirectory / output_filename) as f:
        f["DecayTree"] = df_to_process

    print(f"Successfully saved data with BDT scores to '{OutputDirectory / output_filename}'")