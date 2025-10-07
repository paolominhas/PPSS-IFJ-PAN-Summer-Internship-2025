import uproot
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import PlotStats as plt_stat
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel


def Processing(file_sig, file_bkg, tree_name, variables):
    """
    Function to process ROOT files, extract features, and split data into training and testing sets.
    This function reads signal and background ROOT files, extracts specified variables, and splits the data into training and testing sets.
    It also handles missing values and infinite values, ensuring the data is clean before training.
    
    Parameters:
    file_sig (str): Path to the signal ROOT file, which is a MC simulation of signal.
    file_bkg (str): Path to the background ROOT file.
    tree_name (str): Name of the tree in the ROOT files.
    variables (str): Path to the text file containing variable names of variable used in training.
    
    Returns:
    X_train, X_test, y_train, y_test: Processed training and testing datasets.
    This function also prints the number of events loaded, cleaned, and split into training and testing sets.
    It visualizes the ROC curve and prints the classification report.
    """

    ######################################################
    #                                                    #
    #             Initialise the dataframes:             #
    #                                                    #
    ######################################################

    DataDirectory = Path(__file__).parent.parent / 'InputData'
    features = np.loadtxt(DataDirectory / variables, dtype=str).tolist()
    with uproot.open(DataDirectory / file_sig) as f:
        df_sig = f[tree_name].arrays(features, library="pd")
    with uproot.open(DataDirectory / file_bkg) as f:
        df_bkg_all = f[tree_name].arrays(features + ["Lc_MM"], library="pd")

    df_bkg = df_bkg_all[df_bkg_all["Lc_MM"] > 2310.].copy()  # Apply cut on Lc_MM to remove background events with Lc_MM < 2300 MeV/c^2
    df_bkg.drop(["Lc_MM"], axis=1, inplace=True)   # Remove Lc_MM mass column after cut

    df_sig['target'] = 1    # Signal events labeled as 1
    df_bkg['target'] = 0    # Background events labeled as 0
    df = pd.concat([df_sig, df_bkg], ignore_index=True)
    print(f"Data loaded. Total events: {len(df)}, Signal: {len(df_sig)}, Background: {len(df_bkg)}")

    ######################################################
    #                Cut for non finite                  #
    #                        &                           #
    #             Split test and train data              #   
    ######################################################

    print(f"Original number of events: {len(df)}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)     # Replace infinite values with NaN
    df.dropna(inplace=True)                                 # Drop rows with NaN values
    print(f"Number of events after cleaning: {len(df)}")
    df_X = df[features]        # Features matrix
    df_y = df['target']        # Target vector

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.3, random_state=42, stratify=df_y
    )   # Split data into 70% training and 30% testing sets like TMVA does
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def XGB(X_train, X_test, y_train, y_test):
    """
    Function to build and evaluate an XGBoost model for classification.
    This function performs feature selection, trains the model, evaluates its performance, and visualizes results.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training target labels.
    y_test (pd.Series): Testing target labels.

    Returns:
    None: The function prints the results and plots the ROC curve and feature importances.
    """

    ######################################################
    #                                                    #
    #             Automated Data Selection               #
    #                                                    #
    ######################################################

    
    print("\nPerforming feature selection...")
    prelim_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    prelim_model.fit(X_train, y_train)  # Train preliminary XGBoost model on all features
    selection = SelectFromModel(prelim_model, threshold='median', prefit=True) # pick features with importance above the median
    
    selected_features = X_train.columns[(selection.get_support())]
    selected_feature_names = X_train.columns[(selection.get_support())]
    print(f"Selected {len(selected_features)} features out of {len(X_train.columns)}.")
    X_train_selected = selection.transform(X_train)
    X_test_selected = selection.transform(X_test) # Transform data keep only selected features
    print("Selected features:", selected_features.tolist())
    

    ######################################################
    #                                                    #
    #               Training & Evaluation                #
    #                                                    #
    ######################################################

    final_bdt = XGBClassifier(
        n_estimators=100,                                               # Number of trees (NTrees in TMVA)
        max_depth=3,                                                    # Max depth of each tree  
        learning_rate=0.1,                                              # Learning rate (Shrinkage in TMVA)
        #subsample=0.5,                                                  # Fraction of samples used for fitting each base learner
        #scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),  # Handle class imbalance
        min_child_rate=10,                                              # Minimum child weight (min_child_weight in TMVA)
        use_label_encoder=False,                                        # Avoid warning about label encoder
        eval_metric='logloss',                                          # Evaluation metric            
        random_state=42                                                 # Random state for reproducibility
    )
    final_bdt.fit(X_train_selected, y_train)
    print("Training complete.")
    y_pred_proba = final_bdt.predict_proba(X_test_selected)[:, 1]
    y_pred_class = final_bdt.predict(X_test_selected)

    y_score_test = final_bdt.predict_proba(X_test_selected)[:, 1] 
    plt_stat.Punzi(y_test, y_score_test, n_sigma=5.0)                        # Using 5-sigma

    ######################################################
    #             Visualisation and Plotting             #
    #              (Plotting the ROC Curve)              #
    #             and the Feature Importance             #
    ######################################################

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred_class, target_names=['Background', 'Signal']))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred_class):.4f}")

    plt_stat.pdfplot("XGBoostPlots.pdf", X_train_selected, X_test_selected, y_train, y_test, final_bdt, selected_feature_names)
    plt_stat.SaveBDTResults(final_bdt, selected_feature_names, output_filename="BDTOutputLabelled.root", output_filename_ref="BDTOutputRefLabelled.root")
    #plt_stat.SaveBDTResultsSmall(final_bdt, selected_feature_names, output_filename="BDTOutputSmall.root")


def main():
    print("Starting TMVA-like analysis using XGBoost...")
    XGB(*Processing("Lc2pemu_MC.root", "Lc2pemu_DATA_osign_noBrem.root", "DecayTree", "long_list.txt"))
    #XGB(*Processing("Lc2pphimumu_MC.root", "Lc2pmumu_DATA.root", "DecayTree", "long_list.txt"))
    print("Analysis complete.")
    
if __name__ == "__main__":
    main()