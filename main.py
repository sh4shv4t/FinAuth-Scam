import os
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import the correct, updated functions from your other modules
from model_loader import load_model, preprocess_for_ann
from llm_analyzer import analyze_with_llm # Ensures we are calling the correct function

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# --- Load the pre-trained ANN model at startup ---
try:
    ann_model = load_model('models/fraud_detection_ann.h5')
    print("✅ ANN model loaded successfully at startup.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load the ANN model.")
    print(f"Error details: {e}")
    ann_model = None

@app.route('/analyze', methods=['POST'])
def analyze_transaction():
    """
    Main endpoint to analyze a transaction for potential fraud.
    """
    if ann_model is None:
        return jsonify({"error": "Server is in a degraded state: ANN model is not loaded."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON data received."}), 400

        # --- 1. Validate Input Data ---
        current_transaction = data.get("current_transaction")
        historical_data = data.get("historical_data")
        
        if not current_transaction or not historical_data:
            return jsonify({"error": "Missing required keys: 'current_transaction' or 'historical_data'"}), 400

        # --- 2. Get Scam Score from ANN ---
        processed_input = preprocess_for_ann(current_transaction)
        scam_score = ann_model.predict(processed_input)[0][0]
        scam_score = float(scam_score)
        

        # --- 3. Get Analysis from LLM ---
        llm_response = analyze_with_llm(data, scam_score)

        # --- 4. Format and Return the Final Response ---
        final_response = {
            "scam_score": round(scam_score, 4),
            "llm_analysis": llm_response
        }
        return jsonify(final_response), 200

    except Exception as e:
        # This is a general catch-all for any unexpected crash during processing.
        print(f"❌ An unexpected error occurred in /analyze endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

