
# 💳 Machine Learning-Based Fraud Detection for Operational Risk

## 📘 Project Overview
This project focuses on detecting fraudulent credit card transactions using both **supervised** and **unsupervised machine learning** techniques.  
The goal is to build a robust fraud detection pipeline that not only identifies known patterns of fraud but also detects subtle or novel anomalies to estimate potential **operational risk**.

---

## 🧾 Dataset
- **Source:** Credit Card Transactions Dataset (284,807 rows × 31 columns)
- **Target Variable:** `Class`  
  - `0` → Non-Fraudulent  
  - `1` → Fraudulent  
- **Key Features:** `Time`, `Amount`, and PCA-transformed features (`V1`–`V28`)

---

## 🧠 Comprehensive Analysis Summary

### 1. Data Loading and Initial Exploration
- The dataset was loaded using **pandas** and inspected for missing values and column information.
- Found a severe **class imbalance**:  
  - Non-fraudulent (Class 0): **99.83%**
  - Fraudulent (Class 1): **0.17%**

---

### 2. Data Preprocessing
- **Scaling:**  
  - `Time` and `Amount` were normalized using **StandardScaler**.
- **Outlier Removal:**  
  - Outliers in the `Amount` column were removed using the **Interquartile Range (IQR)** method to reduce the effect of extreme values on model training.

---

### 3. Feature Engineering
- Created a new temporal feature:  
  - `avg_amount_by_time` → mean transaction amount per rounded time interval.
- The new feature was also **scaled** for consistency.

---

### 4. Handling Class Imbalance
- Used **RandomUnderSampler** to balance the dataset.
- Resulted in equal numbers of fraudulent and non-fraudulent samples for supervised learning:
  - Balanced dataset → `X_resampled`, `y_resampled`

---

### 5. Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** to the resampled dataset.
- Retained **95% of the variance**, resulting in a reduced, more compact feature set `X_pca`.

---

### 6. Model Building and Hyperparameter Tuning
- **Models Used:**
  - Random Forest
  - XGBoost
  - Gradient Boosting Machine (GBM)
- Each model was trained on the balanced PCA-transformed training data.
- Used **GridSearchCV** (5-fold) with **F1-score** optimization.
- Best-performing parameter configurations were stored for future use.

---

### 7. Anomaly Detection (Unsupervised)
- Trained an **Isolation Forest** model separately on non-fraudulent transactions.
- The model generated **anomaly scores** for test transactions using the original PCA features (`V1–V28`).
- This provided an unsupervised benchmark for anomaly detection.

---

### 8. Model Evaluation
- Evaluation Metrics:  
  **Precision**, **Recall**, **F1-score**, and **AUC-ROC** (suitable for imbalanced data).
- All three ensemble models achieved **AUC > 0.97**, demonstrating strong discriminatory performance.
- An **ensemble probability average** (XGBoost + GBM) was used for final fraud predictions.

---

### 9. Operational Risk Estimation
- Computed the **average fraudulent transaction amount**.
- Calculated **total estimated operational risk** by summing:
  \[
  \text{Fraud Probability} \times \text{Average Fraud Amount}
  \]
  across all test transactions.
- Provided a quantitative estimate of potential financial loss from undetected fraud.

---

### 10. Insights: Combining Supervised and Unsupervised Approaches
- Compared transactions flagged as:
  - **Ensemble Only**
  - **Ensemble + Isolation Forest**
- Observations:
  - **Both models flagged:** Extreme and variable PCA feature values (clear anomalies)
  - **Ensemble-only flagged:** Subtler patterns in less extreme feature regions
- Insight:
  - **Supervised models** detect known, pattern-based fraud.  
  - **Unsupervised models** capture outliers and novel anomalies.
  - Together, they provide **complementary coverage** of fraudulent behavior.

---

### 11. Strategies for Integration & Future Work
- Proposed methods to combine supervised and unsupervised results:
  - Use Isolation Forest scores as an **additional feature**.
  - Apply **rule-based combinations** or **threshold adjustments**.
  - Build a **meta-ensemble** that weights both prediction sources.
- **Future Enhancements:**
  - Analyze false positives/negatives to refine detection rules.
  - Explore deep learning (e.g., Autoencoders, LSTMs) for temporal fraud patterns.
  - Incorporate transaction metadata (location, device ID, merchant) for richer modeling.

---

## 📊 Tools and Libraries
- **Python 3.10+**
- `pandas`, `numpy`, `scikit-learn`, `imblearn`, `xgboost`, `matplotlib`, `seaborn`

---

## 🏁 Key Takeaways
- Combined **ensemble learning** and **anomaly detection** yields stronger, more reliable fraud detection.
- Feature engineering and imbalance handling are critical for success.
- The operational risk framework provides practical, monetary insights for business risk assessment.

---

## 📂 Folder Structure
```

├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── models/
│   ├── best_xgboost.pkl
│   ├── best_gbm.pkl
│   ├── isolation_forest.pkl
├── results/
│   ├── evaluation_metrics.csv
│   ├── roc_curves.png
│   └── risk_estimation.csv
└── README.md

```

---

**Author:** [Your Name]  
**Date:** October 2025  
**Project:** Machine Learning-Based Fraud Detection for Operational Risk
```

---

Would you like me to generate this as a downloadable **`README.md` file** (Markdown format)?
