import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    page_title="Heart Disease Prediction Portal",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
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

def main():
    st.markdown('<div class="main-title">Heart Disease Diagnostic Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate multiple supervised ML models on clinical test data.</div>', unsafe_allow_html=True)

    st.sidebar.header("Configuration Panel")
    uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
    
    model_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors (kNN)": "knn",
        "Naive Bayes (Gaussian)": "naive_bayes",
        "Random Forest (Ensemble)": "random_forest"
    }
    
    selected_label = st.sidebar.selectbox("Select ML Model", list(model_map.keys()))
    selected_key = model_map[selected_label]
    
    scaler = load_scaler()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
            
        st.subheader("Test Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        
        target_col = "target"
        if target_col not in df.columns:
            st.error(f"Error: Target column '{target_col}' missing from uploaded file.")
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
        
        st.write("---")
        st.subheader(f"Evaluation Metrics: {selected_label}")
        
        cols = st.columns(6)
        metric_names = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
        
        for col, metric in zip(cols, metric_names):
            col.metric(metric, f"{metrics[metric]:.4f}")
            
        st.write("---")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No Disease (0)", "Disease (1)"], 
                        yticklabels=["No Disease (0)", "Disease (1)"])
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            st.pyplot(fig)
            
        with col_right:
            st.subheader("Classification Report")
            # Generate the report as a dictionary
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            
            # Map index labels to understandable names
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
            
            # Display formatted table
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
    else:
        st.info("👈 Please upload `test_data.csv` via the sidebar to start model evaluation.")

if __name__ == "__main__":
    main()