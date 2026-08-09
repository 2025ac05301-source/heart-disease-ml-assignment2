!pip install kagglehub
import os
import shutil
import joblib
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def get_kaggle_dataset():
    data_filename = "heart.csv"
    if not os.path.exists(data_filename):
        print("Downloading dataset using kagglehub...")
        path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
        print("Dataset downloaded to cache path:", path)
        
        # Locate heart.csv inside the downloaded directory structure
        downloaded_csv_path = None
        for root, dirs, files in os.walk(path):
            if "heart.csv" in files:
                downloaded_csv_path = os.path.join(root, "heart.csv")
                break
                
        if downloaded_csv_path and os.path.exists(downloaded_csv_path):
            shutil.copy(downloaded_csv_path, data_filename)
            print("Successfully copied 'heart.csv' to project root directory.")
        else:
            raise FileNotFoundError("Could not locate 'heart.csv' inside the downloaded kagglehub bundle.")

def train_and_export():
    os.makedirs("models", exist_ok=True)
    
    # 1. Fetch data via kagglehub
    get_kaggle_dataset()
    
    df = pd.read_csv("heart.csv")
    print(f"Loaded dataset successfully with shape: {df.shape}")
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # 2. Split dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Export test dataset (test_data.csv) for Streamlit UI
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved 'test_data.csv' containing {len(test_df)} records.")
    
    # 4. Standardize continuous features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "models/scalar.pkl")
    
    # 5. Define the 5 mandatory classification models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    # 6. Fit models and save binary artifacts (.pkl)
    for name, model in models.items():
        if name in ["logistic_regression", "knn", "naive_bayes"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
            
        path = f"models/{name}.pkl"
        joblib.dump(model, path)
        print(f"Trained and saved artifact: {path}")

if __name__ == "__main__":
    train_and_export()