# Heart Disease Risk Prediction using Supervised Binary Classification
---
## BITS Pilani - M.Tech(Artificial Intelligence & Machine Learning)
---
## Machine Learning(AIMLCZG565 - Assignment 2)
---
## Problem statement
Cardiovascular diseases (CVDs) are a primary cause of global mortality. Early, automated screening of patient parameters helps clinicians detect underlying coronary risks efficiently. 

This project implements, trains, and compares 5 core supervised classification models using data pulled from the public repository - Kaggle. An interactive Streamlit dashboard is built and deployed to facilitate live model inspection and metric validation.

---

## Dataset description
The dataset is acquired directly from Kaggle (`johnsmith88/heart-disease-dataset`).

* **Source:** Kaggle Repository (`johnsmith88/heart-disease-dataset`)
* **Total Instances:** 1,025 records (820 training / 205 test split)
* **Total Features:** 13 input clinical variables + 1 target column (`target`)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age in years |
| `sex` | Categorical | Sex (1 = male; 0 = female) |
| `cp` | Categorical | Chest pain type (4 values) |
| `trestbps` | Numerical | Resting blood pressure (mm Hg) |
| `chol` | Numerical | Serum cholesterol in mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting electrocardiographic results |
| `thalach` | Numerical | Maximum heart rate achieved |
| `exang` | Categorical | Exercise induced angina (1 = yes; 0 = no) |
| `oldpeak` | Numerical | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Numerical | Number of major vessels colored by fluoroscopy |
| `thal` | Categorical | Thalassemia (1 = normal; 2 = fixed; 3 = reversible) |

---

## Github Repository Link
* **GitHub Repository:** `https://github.com/2025ac05301-source/heart-disease-ml-assignment2`

---

## Models used

Computed on held-out test data (`test_data.csv`, $N = 205$).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7805 | 0.8738 | 0.8000 | 0.7619 | 0.7805 | 0.5619 |
| **Decision Tree** | 0.7805 | 0.7798 | 0.7727 | 0.8095 | 0.7907 | 0.5609 |
| **kNN** | 0.7561 | 0.8810 | 0.7391 | 0.8095 | 0.7727 | 0.5132 |
| **Naive Bayes (Gaussian)** | 0.7805 | 0.8905 | 0.7727 | 0.8095 | 0.7907 | 0.5609 |
| **Random Forest (Ensemble)** | 0.8049 | 0.8940 | 0.8095 | 0.8095 | 0.8095 | 0.6095 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Stable linear baseline model. Scales well with normalized features, yielding a high AUC of 0.8738 and solid precision (0.8000), though bounded by linear separation constraints. |
| **Decision Tree** | Handles multi-split feature dependencies well, achieving 78.05% accuracy and an F1 score of 0.7907, capturing non-linear clinical thresholds cleanly. |
| **kNN** | Performs soundly with distance metrics calculated on scaled inputs, achieving 75.61% accuracy and a strong AUC of 0.8810. |
| **Naive Bayes (Gaussian)** | Executes rapidly and captures a strong AUC of 0.8905 and recall (0.8095), balancing probabilistic conditional independence. |
| **Random Forest (Ensemble)** | **Overall Winner.** Ensembling 100 decorrelated trees heavily suppresses overfitting, capturing peak scores across all dimensions (Accuracy: 0.8049, AUC: 0.8940, Recall: 0.8095, F1: 0.8095, MCC: 0.6095). |

#### **Overall Winner for heart disease dataset:** **Random Forest (Ensemble)**