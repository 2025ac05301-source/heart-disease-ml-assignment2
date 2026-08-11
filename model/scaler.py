from sklearn.preprocessing import StandardScaler

def get_scaler(X_train):
    """Fits and returns the standard scaler using training features only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler