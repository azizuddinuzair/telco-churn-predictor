# Customer Churn – Observations & Experiments

## Raw MI (before engineering)
- customerID: 0.5786 (drop; target leakage/id-like)
- TotalCharges: 0.5410 (inflated due to categorical/high-cardinality treatment; later fixed)
- Contract: 0.0989 
- tenure: 0.0779 
- OnlineSecurity: 0.0661 
- TechSupport: 0.0638 
- InternetService: 0.0570 
- OnlineBackup: 0.0475 
- PaymentMethod: 0.0469 
- DeviceProtection: 0.0436 
- MonthlyCharges: 0.0411 
- StreamingMovies: 0.0323 
- StreamingTV: 0.0321
- PaperlessBilling: 0.0205
- others ~0

## Engineered MI
- Contract_Tenure: 0.1109
- OnlineSecurity_TechSupport: 0.0857
- PaperlessBilling_PaymentMethod: 0.0587
- InternetService: 0.0570
- OnlineBackup_DeviceProtection: 0.0555
- MonthlyCharges: 0.0467
- TotalCharges (numeric): 0.0400
- StreamingMovies: 0.0323
- StreamingTV: 0.0321


Notes
- Kept: Contract_Tenure (+0.02) 
- OnlineSecurity_TechSupport (+0.05), 
- OnlineBackup_DeviceProtection (+0.04), 
- PaperlessBilling_PaymentMethod (small but simplifies data)
- Attempted and dropped combos (<0.01 gain): Dependents_Partner, StreamingMovies_StreamingTV
- TotalCharges inflated when treated categorical; normalized when changed to numeric

## Model Experiments

### Experiment 1 (baseline)
| Model | Val F1 | Val Acc | CV F1 | CV Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC | 0.5351 | 0.7743 | 0.5094 | 0.7677 | Mild overfit |
| KNN | 0.5774 | 0.7154 | 0.3235 | 0.7354 | Strong overfit |
| LR  | 0.5325 | 0.7807 | 0.5569 | 0.7964 | Stable baseline |

### Experiment 2 (RFC min_samples_leaf=7; KNN weights=distance)
| Model | Val F1 | Val Acc | CV F1 | CV Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC | 0.5653 | 0.7850 | 0.5799 | 0.7930 | Good generalization |
| KNN | 0.5604 | 0.7161 | 0.4253 | 0.7485 | Still overfit |
| LR  | 0.5325 | 0.7807 | 0.5569 | 0.7964 | Stable |

### Experiment 3 (RFC max_depth=10, n_estimators=200; KNN metric=manhattan)
| Model | Val F1 | Val Acc | CV F1 | CV Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC | 0.5731 | 0.7885 | 0.5857 | 0.7968 | Best overall |
| KNN | 0.5557 | 0.7651 | 0.4772 | 0.7749 | Overfit reduced |
| LR  | 0.5325 | 0.7807 | 0.5569 | 0.7964 | Stable |

### Experiment 4 (Fixed TotalCharges numeric)
| Model | Val F1 | Val Acc | CV F1 | CV Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC | 0.5698 | 0.7921 | 0.5694 | 0.7959 | Stable |
| KNN | 0.5110 | 0.7488 | 0.5421 | 0.7650 | Slight underfit |
| LR  | 0.5569 | 0.7842 | 0.5856 | 0.7993 | Near RFC |

## Takeaways
- Drop customerID; changed TotalCharges to numeric to avoid inflated MI from high-cardinality categories.
- Best combos: Contract_Tenure, OnlineSecurity_TechSupport, OnlineBackup_DeviceProtection, PaperlessBilling_PaymentMethod.
- RFC is most stable and best overall; LR is a strong, simple baseline; KNN remains weakest.
- TotalCharges adds modest signal once numeric; binning didn’t help.
