
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(scale=True):

    # Load dataset
    dataset = load_breast_cancer()

    X = dataset.data
    y = (dataset.target == 0).astype(np.float32)

    # Split: 60% training, 40% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # Split temporary data into 20% validation and 20% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    scaler = None

    # Optional scaling
    if scale:
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler
    )
