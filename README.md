# Summer Internship - IFJ PAN

I worked on charmed baryon decays at the IFJ PAN laboratory in Krakow. Issues with storage on my laptop (being now completely full) and the size of the data files has meant I have not been able to turn the code into a standard repository on my mashine and then push to GitHub. After putting this all on a drive I hope to be able to upload the project and report onto here properly. For now please see the results I have put here as well as some of the testing and training python files, as well as the ROOT version woth TMVA as well as some of the notes I took on the project and the presentation made with Timur Knyazev at the end of the programme. Thank you to Prof. Dr. Hab. Mariusz Witek for his help with this project.

For details of what we were doing see the [notes](./notes).




# Information about these files:

## Settings
These two ntuples were calulated using the following settings:

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

## Training Log:

Starting TMVA-like analysis using XGBoost...
Data loaded. Total events: 80855, Signal: 1727, Background: 79128
Original number of events: 80855
Number of events after cleaning: 80855
Training set size: 56598, Testing set size: 24257

Performing feature selection...

Selected features: ['Lc_IPCHI2_OWNPV', 'Lc_FD_OWNPV', 'Lc_transformed_TAU', 'Lc_DIRA_OWNPV', 'Lc_TAUCHI2', 'Lc_FITPV_PVLTIME', 'p_PT', 'p_uniProbNNp', 'DTF_CHI2', 'p_FITPV_IPCHI2', 'l_PT', 'mu_PT', 'p_FITPV_IP', 'l_FITPV_PT', 'p_FITPV_PT', 'mu_FITPV_PT', 'relinfo_VTXISOBDTHARDFIRSTVALUE', 'relinfo_TrackIsoBDTp_TRKISOBDTTHIRDVALUE']
Parameters: { "min_child_rate", "use_label_encoder" } are not used.

Training complete.
Total signal events: 518, Total background events: 23739

--- Punzi Optimization ---
Optimal BDT score cut: 0.2456
Maximum Punzi Significance: 0.0203

Classification Report on Test Set:
              precision    recall  f1-score   support

  Background       0.98      1.00      0.99     23739
      Signal       0.70      0.16      0.26       518

    accuracy                           0.98     24257
   macro avg       0.84      0.58      0.62     24257
weighted avg       0.98      0.98      0.97     24257

Accuracy Score: 0.9806

## Decay tree components

```
paolo_xgb_files/
├── BDTOutputLabelled.root
│   └── (The signal data and background combined)
├── BDTOutputRefLabelled.root
│   └── (Reference channel data)
├── XGBoostPlots.pdf
└── Punzi_Significance.pdf
```

- **`BDTOutputLabelled.root`**: This directory contains all the input data files required for the analysis.
  - `DecayTree`: The tree name.
    - `target`: The labelling of MC and DATA (as 1 and 0 respectively).
    - `bdt_score`: The score (a double from 0 to 1) predicted by the BDT model

- **`BDTOutputRefLabelled.root`**: This directory contains all the Python source code.
  - `DecayTreeRef`: The tree name.
    - `target`: The labelling of MC and DATA (as 1 and 0 respectively).
    - `bdt_score`: The score (a double from 0 to 1) predicted by the BDT model

- The other files are plots showing the model performanc in this particular run.
