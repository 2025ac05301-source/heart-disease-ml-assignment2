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
| **Logistic Regression** | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| **Decision Tree** | 0.8732 | 0.9326 | 0.8624 | 0.8952 | 0.8785 | 0.7465 |
| **kNN** | 0.8439 | 0.9453 | 0.8230 | 0.8857 | 0.8532 | 0.6891 |
| **Naive Bayes (Gaussian)** | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| **Random Forest (Ensemble)** | 0.9610 | 0.9863 | 0.9533 | 0.9714 | 0.9623 | 0.9220 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Stable linear baseline model. Scales well with normalized features, yielding a high AUC of 0.9298 and strong recall (0.9143), though bounded by linear separation constraints. |
| **Decision Tree** | Handles multi-split feature dependencies well, achieving 87.32% accuracy and an F1 score of 0.8785, capturing non-linear clinical thresholds cleanly. |
| **kNN** | Performs soundly with distance metrics calculated on scaled inputs, achieving 84.39% accuracy and a strong AUC of 0.9453. |
| **Naive Bayes (Gaussian)** | Executes rapidly and captures solid recall (0.8762) and AUC (0.9043), though feature distribution assumptions slightly lower precision. |
| **Random Forest (Ensemble)** | **Overall Winner.** Ensembling 100 decorrelated trees heavily suppresses overfitting, capturing exceptional scores across all dimensions (Accuracy: 0.9610, AUC: 0.9863, Recall: 0.9714, F1: 0.9623, MCC: 0.9220). |

#### **Overall Winner for heart disease dataset:** **Random Forest (Ensemble)**