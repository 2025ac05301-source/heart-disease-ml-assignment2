import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Heart Disease Intelligence Portal",
    page_icon="🩺",
    layout="wide"
)

# Additional CSS Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("models/scalar.pkl")
    except Exception:
        return None

@st.cache_resource
def load_model(model_key):
    try:
        return joblib.load(f"models/{model_key}.pkl")
    except Exception:
        return None

def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = 0.5
        
    return {
        "Accuracy": acc,
        "AUC Score": auc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "MCC Score": mcc
    }
    
def generate_runtime_observation(metrics, selected_label):
    acc = metrics["Accuracy"]
    f1 = metrics["F1 Score"]
    auc = metrics["AUC Score"]
    mcc = metrics["MCC Score"]
    
    # Dynamically categorize performance tier based on live runtime metrics
    if f1 >= 0.90 and mcc >= 0.80:
        tier = "exceptional high-tier performance"
    elif f1 >= 0.80:
        tier = "strong and robust operational performance"
    else:
        tier = "moderate baseline performance"
        
    observation = (
        f"**Evaluation Insight for {selected_label}:** "
        f"Operating on the currently loaded dataset, this model demonstrates {tier}. "
        f"It achieved an overall Accuracy of **{acc:.4f}** and an F1-Score of **{f1:.4f}**, "
        f"complemented by an AUC-ROC rating of **{auc:.4f}** and an MCC score of **{mcc:.4f}**. "
    )
    
    if selected_label == "Random Forest (Ensemble)":
        observation += "Ensembling multiple decision trees effectively minimizes variance and stabilizes predictions across target classes."
    elif selected_label == "Logistic Regression":
        observation += "Linear boundary separation handles scaled input parameters stably, maintaining steady generalization."
    elif selected_label == "Decision Tree":
        observation += "Recursive splits capture non-linear feature interactions directly from the dataset features."
    elif selected_label == "K-Nearest Neighbors (kNN)":
        observation += "Distance-based feature weighting correctly classifies neighboring instances based on normalized feature space."
    elif selected_label == "Naive Bayes (Gaussian)":
        observation += "Probabilistic conditional independence calculations execute rapidly while preserving class boundary estimates."
        
    return observation

def main():
    # App Title & Overview
    st.markdown('<div class="main-title">🩺 Heart Disease Intelligence Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced clinical evaluation dashboard powered by supervised machine learning models.</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### ⚙️ Control Panel")
    st.sidebar.markdown("Configure your dataset and algorithm parameters below.")
    
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
    
    if not csv_files:
        st.sidebar.error("No CSV files found in the repository root directory.")
        return
        
    default_index = 0
    if "test_data.csv" in csv_files:
        default_index = csv_files.index("test_data.csv")
        
    selected_csv = st.sidebar.selectbox("📂 Evaluation Dataset", csv_files, index=default_index)
    
    model_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors (kNN)": "knn",
        "Naive Bayes (Gaussian)": "naive_bayes",
        "Random Forest (Ensemble)": "random_forest"
    }
    
    selected_label = st.sidebar.selectbox("🔬 Supervised Algorithm", list(model_map.keys()))
    selected_key = model_map[selected_label]
    
    scaler = load_scaler()
    
    try:
        df = pd.read_csv(selected_csv)
    except Exception as e:
        st.error(f"Error reading dataset file '{selected_csv}': {e}")
        return
        
    # Dataset Preview Section
    with st.expander(f"🔍 Preview Dataset Schema & Records ({selected_csv})", expanded=True):
        st.dataframe(df.head(6), use_container_width=True)
    
    target_col = "target"
    if target_col not in df.columns:
        st.error(f"Error: Target column '{target_col}' missing from selected dataset.")
        return
        
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    model = load_model(selected_key)
    if model is None:
        st.error(f"Failed to load model file for: {selected_label}")
        return
        
    if selected_key in ["logistic_regression", "knn", "naive_bayes"] and scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values if hasattr(X_test, "values") else X_test

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
        
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Section 1: Performance Metrics Grid
    st.markdown("---")
    st.markdown(f"### 📊 Key Performance Indicators: **{selected_label}**")
    
    cols = st.columns(6)
    metric_names = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
    
    for col, metric in zip(cols, metric_names):
        with col:
            st.metric(label=metric, value=f"{metrics[metric]:.4f}")

    # Display Performance Observation Box
    runtime_observation = generate_runtime_observation(metrics, selected_label)
    st.info(runtime_observation)
            
    # Section 2: Side-by-Side Deep Dive Visualizations (Confusion Matrix & Classification Report)
    st.markdown("---")
    st.markdown("### 🔍 Model Diagnostics & Breakdown")
    
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    with col_left:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Disease (0)", "Disease (1)"], 
                    yticklabels=["No Disease (0)", "Disease (1)"])
        ax.set_xlabel("Predicted Diagnosis", fontweight="bold")
        ax.set_ylabel("Actual Diagnosis", fontweight="bold")
        st.pyplot(fig)
        
    with col_right:
        st.markdown("#### Classification Report Breakdown")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        index_mapping = {
            "0": "Class 0 - No Heart Disease",
            "1": "Class 1 - Heart Disease Present",
            "accuracy": "Accuracy",
            "macro avg": "Macro Average",
            "weighted avg": "Weighted Average"
        }
        report_df = report_df.rename(index=index_mapping)
        
        column_mapping = {
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Total Instances"
        }
        report_df = report_df.rename(columns=column_mapping)
        
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

if __name__ == "__main__":
    main()