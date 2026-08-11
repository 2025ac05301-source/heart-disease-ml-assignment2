from sklearn.naive_bayes import GaussianNB

def train_model(X_train_scaled, y_train):
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    return model