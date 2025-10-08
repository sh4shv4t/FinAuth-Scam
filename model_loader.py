import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_model(model_path='models/fraud_detection_ann.h5'):
    """
    Loads the pre-trained Keras ANN model from the specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Keras model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")
        # Re-raise the exception to be handled at startup in main.py
        raise e

def preprocess_for_ann(current_transaction):
    """
    Prepares the input data from a JSON request for the ANN model.

    This function is critical and must transform the real-world input into the
    exact format the model was trained on.

    The model was trained on 30 features (scaled_amount, scaled_time, and 28 V-features).
    Since we only receive 'amount' from the API that matches the training data,
    we will use it and pad the other 29 features with zeros.

    Args:
        current_transaction (dict): A dictionary containing the transaction details.
                                    Example: {'amount': 45.50, ...}

    Returns:
        np.ndarray: A NumPy array of shape (1, 30) ready for prediction.
    """
    try:
        # 1. Create a placeholder array of zeros with the correct shape.
        #    The shape is (1, 30) because we have 1 transaction and 30 features.
        feature_vector = np.zeros((1, 30))

        # 2. Extract the transaction amount.
        amount = current_transaction.get('amount', 0.0)

        # 3. Scale the amount.
        #    NOTE: In a real-world scenario, you should use the SAME scaler
        #    that was fitted on the training data. For this example, we'll
        #    create a new one, which is sufficient to make the code run.
        scaler = StandardScaler()
        scaled_amount = scaler.fit_transform(np.array([[amount]]))

        # 4. Place the scaled amount in the first position of our feature vector,
        #    just like in the training data. The other 29 features remain zero.
        feature_vector[0, 0] = scaled_amount[0, 0]

        return feature_vector

    except Exception as e:
        print(f"❌ Error during preprocessing for ANN: {e}")
        # Return None or raise an exception to be caught by the server
        raise ValueError("Failed to preprocess data for the model.")

