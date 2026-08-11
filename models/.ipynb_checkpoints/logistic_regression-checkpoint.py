from sklearn.linear_model import LogisticRegression

def train_model(X_train_scaled, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model