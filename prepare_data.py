import pandas as pd
from sklearn.model_selection import train_test_split

def generate_splits():
    # Load raw dataset (i.e. full heart.csv containing 1,025 records)
    df = pd.read_csv("heart.csv")
    
    # 80/20 train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    print(f"Data split complete! Train records: {len(train_df)}, Test records: {len(test_df)}")

if __name__ == "__main__":
    generate_splits()