#  Food Delivery Status Prediction: Model Comparison and Actionable Insights

## Phase 3: Reporting and Insights

This phase concludes the project by comparing the performance of the three binary classification models and translating the results into actionable business strategies for improving delivery logistics.

---

##  Model Comparison and Evaluation

We evaluated three machine learning classifiers—**Gaussian Naive Bayes (GNB)**, **K-Nearest Neighbors (KNN)**, and **Tuned Decision Tree (DT)**—on their ability to predict the binary target: **Delivery Status (Fast/Delayed)**.

### 1. Performance Metrics Summary

| Model | Hyperparameter | Accuracy | Precision (Delayed: 1) | Recall (Delayed: 1) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes (GNB)** | N/A | **[Insert GNB Accuracy]** | **[Insert GNB Precision]** | **[Insert GNB Recall]** | **[Insert GNB F1-Score]** |
| **KNN** | K = **[Insert Optimal K]** | **[Insert KNN Accuracy]** | **[Insert KNN Precision]** | **[Insert KNN Recall]** | **[Insert KNN F1-Score]** |
| **Decision Tree (DT)** | Max Depth = **[Insert DT Max Depth]** | **[Insert DT Accuracy]** | **[Insert DT Precision]** | **[Insert DT Recall]** | **[Insert DT F1-Score]** |

*(**Note:** Precision measures how many predicted "Delayed" cases were actually delayed. Recall measures how many actual "Delayed" cases were correctly identified.)*

### 2. Visualization Code

To generate the comparative visualizations required for the report:

```python
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_test, y_test, and trained models (gnb_model, knn_model, dt_model) are available.

# --- 1. Combined ROC Curve Plot ---
plt.figure(figsize=(8, 6))

# Define classifiers and their predictions (using predict_proba[:, 1] for ROC)
classifiers = [
    (gnb_model, 'Gaussian Naive Bayes'),
    (knn_model, f'KNN (K={knn_model.n_neighbors})'),
    (dt_model, f'Decision Tree (Depth={dt_model.max_depth}')
]

# Plot ROC for each model
for model, name in classifiers:
    # Use CalibratedClassifierCV for models that lack robust probability output (like some DTs/KNNs)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Fallback for models without predict_proba (less common)
        calibrated_model = CalibratedClassifierCV(model, cv="prefit")
        calibrated_model.fit(X_train, y_train)
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc="lower right")
plt.show()
# 
