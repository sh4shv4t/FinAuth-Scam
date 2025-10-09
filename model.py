# ANN Model Training for Credit Card Fraud Detection
#
# This script walks through the complete process of training an Artificial Neural
# Network (ANN) to detect fraudulent credit card transactions. It uses the
# popular Credit Card Fraud Detection dataset from Kaggle.
#
# The process is divided into five main steps:
# 1.  Setup and Data Download: Set up the Kaggle API and download the dataset.
# 2.  Data Preprocessing: Load, explore, and prepare the data for the model.
# 3.  ANN Model Building: Define the architecture of the neural network.
# 4.  Model Training and Evaluation: Train the model, handling class imbalance,
#     and then evaluate its performance.
# 5.  Saving the Model: Save the final trained model for use in the backend.

import os
import zipfile
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Setup and Data Download ---
#
# First, we need to ensure the necessary libraries are installed and set up the
# Kaggle API to download our dataset directly.
#
# Important: Before running this script, ensure you have placed your `kaggle.json`
# API token file in the correct directory (`~/.kaggle/` on Linux/macOS or
# `C:\\Users\\<Your-Username>\\.kaggle\\` on Windows).

def setup_and_download_data():
    """
    Ensures directories exist, downloads, and unzips the dataset from Kaggle.
    """
    # Ensure the .kaggle directory exists using exist_ok=True to prevent errors.
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)

    # Create the data directory, also with exist_ok=True
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Download and unzip the dataset using the Kaggle CLI
    print("Downloading dataset from Kaggle...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", data_dir,
            "--unzip"
        ], check=True)
        print("Dataset downloaded and extracted successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n--- KAGGLE CLI ERROR ---")
        print("Could not download the dataset. Please ensure the following:")
        print("1. You have run 'pip install kaggle'.")
        print("2. Your 'kaggle.json' file is correctly placed in the ~/.kaggle/ directory.")
        print(f"Error details: {e}")
        exit() # Exit the script if data download fails

# --- Step 2: Load and Preprocess the Data ---
#
# Now that we have the data, we'll load it into a pandas DataFrame. The most
# critical preprocessing step here is to scale the `Time` and `Amount` columns,
# as their values are not on the same scale as the other anonymized PCA features.

def load_and_prepare_data(filepath='data/creditcard.csv'):
    """
    Loads the dataset and prepares it for training.
    """
    try:
        df = pd.read_csv(filepath)
        print("\nDataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")

        # Scale 'Amount' and 'Time' features
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df = df.drop(['Time', 'Amount'], axis=1)

        # Reorder columns to have the scaled features at the beginning
        df.insert(0, 'scaled_amount', df.pop('scaled_amount'))
        df.insert(1, 'scaled_time', df.pop('scaled_time'))

        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Was the download successful?")
        return None

# --- Step 3: Build the ANN Model ---
#
# Here, we define the architecture for our neural network. It's a simple
# sequential model with a few dense layers and dropout layers to prevent
# overfitting. The final layer uses a `sigmoid` activation function, which is
# perfect for binary classification as it outputs a probability between 0 and 1.

def build_fraud_detection_model(input_shape):
    """
    Builds and compiles the ANN model architecture.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# --- Step 4: Train and Evaluate the Model ---
#
# This is the most crucial step. We will train the model, but we need to handle
# the severe class imbalance in the dataset. We'll use `class_weight` to tell
# the model to pay significantly more attention to the fraudulent transactions.

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Trains the model and evaluates its performance.
    """
    # Handle class imbalance by calculating class weights
    neg, pos = np.bincount(y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f"\nClass weights: Fraudulent (1): {weight_for_1:.2f}, Non-Fraudulent (0): {weight_for_0:.2f}")

    # Set up early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    print("\nStarting model training...")
    model.fit(
        X_train,
        y_train,
        batch_size=2048,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1
    )

    print("\n--- Model Evaluation ---")
    predictions_prob = model.predict(X_test)
    predictions = (predictions_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Legitimate', 'Fraud']))

    print("ROC AUC Score:", roc_auc_score(y_test, predictions_prob))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

# --- Step 5: Save the Trained Model ---
#
# Finally, we'll save our trained model to a file named `fraud_detection_ann.h5`.
# This file can then be loaded by our Flask backend to make live predictions.

def save_trained_model(model, path='models/'):
    """
    Saves the final trained model to the specified path.
    """
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, 'fraud_detection_ann.h5')
    model.save(model_path)
    print(f"\nModel saved successfully to {model_path}")

# --- Main Execution Block ---
def main():
    """
    Main function to run the entire model training pipeline.
    """
    # Step 1
    setup_and_download_data()

    # Step 2
    transaction_df = load_and_prepare_data()
    if transaction_df is None:
        return # Exit if data loading failed

    X = transaction_df.drop('Class', axis=1)
    y = transaction_df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nData preprocessing and splitting complete.")

    # Step 3
    input_shape = X_train.shape[1]
    fraud_classifier_model = build_fraud_detection_model(input_shape)
    print("\nANN model built successfully.")
    fraud_classifier_model.summary()

    # Step 4
    train_and_evaluate(fraud_classifier_model, X_train, y_train, X_test, y_test)

    # Step 5
    save_trained_model(fraud_classifier_model)

if __name__ == "__main__":
    main()

