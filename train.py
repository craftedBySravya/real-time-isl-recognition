# train.py
# This script trains the gesture model using the CSV files we collected earlier.

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from config import DATASET_DIR, MODEL_PATH, ENCODER_PATH


def load_all_gesture_csvs(dataset_dir):
    """
    Loads all gesture CSV files from the dataset folder.
    Each CSV has hand landmark values and the label at the end.
    """
    all_data = []

    for fname in os.listdir(dataset_dir):
        if fname.lower().endswith(".csv"):
            path = os.path.join(dataset_dir, fname)
            print(f"Loading: {path}")
            df = pd.read_csv(path, header=None)   # No headers in our CSV files
            all_data.append(df)

    if not all_data:
        raise ValueError("No CSV files in dataset folder.")

    # Combine everything into one big dataframe
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples loaded: {len(df)}\n")

    # Split into features and labels
    X = df.iloc[:, :-1].values.astype("float32")  # all columns except last
    y = df.iloc[:, -1].values                      # last column is label

    return X, y


def augment_data(X, y, copies=2, noise_std=0.01, scale_range=(0.97, 1.03)):
    """
    Makes extra noisy copies of training data to help the model generalize.
    Basically the same data but slightly scaled and with small noise added.
    """
    X_aug = [X]
    y_aug = [y]

    n, d = X.shape

    for _ in range(copies):
        # Random scaling + noise
        scale = np.random.uniform(scale_range[0], scale_range[1], size=(n, 1))
        noise = np.random.normal(0, noise_std, size=(n, d))
        X_new = X * scale + noise

        X_aug.append(X_new)
        y_aug.append(y)

    # Combine everything
    X_final = np.vstack(X_aug)
    y_final = np.concatenate(y_aug)

    return X_final, y_final


def main():

    print("=== TRAINING MODEL ===\n")

    # Step 1: read all data from CSV files
    X, y_raw = load_all_gesture_csvs(DATASET_DIR)

    # Step 2: convert text labels (HELLO, THANK YOU, etc.) into numbers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Classes found:", list(le.classes_), "\n")

    # Step 3: split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    # Step 4: augment the training data to make the model stronger
    X_train_aug, y_train_aug = augment_data(
        X_train, y_train,
        copies=2,
        noise_std=0.015,
        scale_range=(0.95, 1.05)
    )

    print(f"After augmentation: {len(X_train_aug)} samples\n")

    # Step 5: Build the model
    # First scale the data → then train MLP (neural network)
    model = Pipeline([
        ("scaler", StandardScaler()),

        # Neural network with two hidden layers
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=400,
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train_aug, y_train_aug)

    # Step 6: Test how good the model is
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Step 7: Save the trained model so we can use it in app_chat.py
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {ENCODER_PATH}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
