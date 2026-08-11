import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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

# Import modular model components from model/ directory
from model.scaler import get_scaler
from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest

st.set_page_config(
    page_title="Heart Disease Intelligence Portal",
    page_icon="🩺",
    layout="wide"
)

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
def load_and_split_data_and_models(source_df):
    """
    Takes a DataFrame, performs an 80/20 train-test split,
    fits the scaler on training data, and trains individual model scripts.
    """
    try:
        target_col = "target"
        if target_col not in source_df.columns:
            return None, None, None, None, f"Target column '{target_col}' missing from dataset."

        X = source_df.drop(columns=[target_col])
        y = source_df[target_col]
        
        # Explicit 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit scaler on training data
        scaler = get_scaler(X_train)
        X_train_scaled = scaler.transform(X_train)
        
        # Train individual model scripts
        models = {
            "logistic_regression": logistic_regression.train_model(X_train_scaled, y_train),
            "decision_tree": decision_tree.train_model(X_train, y_train),
            "knn": knn.train_model(X_train_scaled, y_train),
            "naive_bayes": naive_bayes.train_model(X_train_scaled, y_train),
            "random_forest": random_forest.train_model(X_train, y_train)
        }
        
        return models, scaler, X_test, y_test, None
    except Exception as e:
        return None, None, None, None, str(e)

def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5
        
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
    
    tier = "exceptional high-tier performance" if f1 >= 0.90 and mcc >= 0.80 else "strong operational performance"
    
    return (
        f"**Runtime Evaluation Insight for {selected_label}:** "
        f"Operating on the evaluation split, this model demonstrates {tier}. "
        f"It achieved an overall Accuracy of **{acc:.4f}** and an F1-Score of **{f1:.4f}**, "
        f"complemented by an AUC-ROC rating of **{auc:.4f}** and an MCC score of **{mcc:.4f}**."
    )

def main():
    st.markdown('<div class="main-title">🩺 Heart Disease Intelligence Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced clinical evaluation dashboard powered by supervised machine learning models.</div>', unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Control Panel")
    
    # Dataset upload option (CSV)
    st.sidebar.markdown("---")
    st.sidebar.markdown("📂 **Dataset Source**")
    uploaded_file = st.sidebar.file_uploader("Upload test data CSV", type=["csv"])
    
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
    
    # Load dataset either from user upload or fallback to local files
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            source_label = f"Uploaded File ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        if not csv_files:
            st.sidebar.error("No CSV files found in root directory.")
            return
            
        default_index = csv_files.index("test_data.csv") if "test_data.csv" in csv_files else 0
        selected_csv = st.sidebar.selectbox("Select Repository Dataset", csv_files, index=default_index)
        try:
            df = pd.read_csv(selected_csv)
            source_label = selected_csv
        except Exception as e:
            st.error(f"Error reading dataset file '{selected_csv}': {e}")
            return

    model_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors (kNN)": "knn",
        "Naive Bayes (Gaussian)": "naive_bayes",
        "Random Forest (Ensemble)": "random_forest"
    }
    
    st.sidebar.markdown("---")
    selected_label = st.sidebar.selectbox("🔬 Supervised Algorithm", list(model_map.keys()))
    selected_key = model_map[selected_label]
    
    models_dict, scaler, X_test, y_test, error = load_and_split_data_and_models(df)
    if error:
        st.error(f"Error loading pipeline: {error}")
        return
    
    with st.expander(f"🔍 Preview Evaluation Split Records ({len(X_test)} instances from {source_label})", expanded=True):
        preview_df = X_test.copy()
        preview_df["target"] = y_test
        st.dataframe(preview_df, use_container_width=True)
    
    model = models_dict.get(selected_key)
    if selected_key in ["logistic_regression", "knn", "naive_bayes"] and scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
        
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    st.markdown("---")
    st.markdown(f"### 📊 Key Performance Indicators: **{selected_label}**")
    
    cols = st.columns(6)
    metric_names = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
    for col, metric in zip(cols, metric_names):
        col.metric(label=metric, value=f"{metrics[metric]:.4f}")
            
    st.info(generate_runtime_observation(metrics, selected_label))
            
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
        
        report_df = report_df.rename(index={
            "0": "Class 0 - No Heart Disease",
            "1": "Class 1 - Heart Disease Present",
            "accuracy": "Accuracy",
            "macro avg": "Macro Average",
            "weighted avg": "Weighted Average"
        })
        report_df = report_df.rename(columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Total Instances"
        })
        
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

if __name__ == "__main__":
    main()