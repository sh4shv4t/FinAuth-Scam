import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime # Import the datetime library

def load_model(model_path='models/fraud_detection_ann.h5'):
    """
    Loads the pre-trained Keras ANN model from the specified path.
    (This function is correct and remains unchanged.)
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Keras model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")
        raise e

def preprocess_for_ann(current_transaction):
    """
    MODIFIED: Prepares input data from a full transaction object for the ANN.
    The model was trained on 30 features (scaled_time, scaled_amount, and 28 V-features).
    This function now extracts and scales both time and amount to better match the training data.
    """
    try:
        # 1. Create a placeholder array of zeros with the correct shape (1, 30).
        feature_vector = np.zeros((1, 30))

        # --- 2. MODIFIED: Extract and process both Amount and Timestamp ---

        # A) Process Amount
        amount = current_transaction.get('amount', 0.0)
        # In a real system, you would load and use a scaler fitted on the original training data.
        amount_scaler = StandardScaler()
        scaled_amount = amount_scaler.fit_transform(np.array([[amount]]))

        # B) Process Timestamp
        timestamp_str = current_transaction.get('timestamp', datetime.utcnow().isoformat())
        # Convert the ISO format string (e.g., "2025-10-09T14:30:00Z") to a Unix timestamp.
        # This numeric value is equivalent to the 'Time' feature in the training data.
        time_unix = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
        time_scaler = StandardScaler()
        scaled_time = time_scaler.fit_transform(np.array([[time_unix]]))

        # --- 3. MODIFIED: Place scaled features into the vector ---
        # We place scaled_time and scaled_amount into the first two positions.
        # The remaining 28 features (representing V1-V28) are left as zeros.
        feature_vector[0, 0] = scaled_time[0, 0]
        feature_vector[0, 1] = scaled_amount[0, 0]

        return feature_vector

    except Exception as e:
        print(f"❌ Error during preprocessing for ANN: {e}")
        raise ValueError("Failed to preprocess data for the model.")

