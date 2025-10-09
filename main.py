import os
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Imports for Location Analysis
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Import your existing custom modules
from model_loader import load_model, preprocess_for_ann
from llm_analyzer import analyze_with_llm

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# --- Initialize Geocoder (Kept for potential future use, though not used by the updated endpoint) ---
geolocator = Nominatim(user_agent="fraud_detection_app_v1")

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
    MODIFIED to handle the new detailed transaction schema.
    """
    if ann_model is None:
        return jsonify({"error": "Server is in a degraded state: ANN model is not loaded."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON data received."}), 400

        # --- 1. MODIFIED: Validate Input Data based on new schema ---
        current_transaction = data.get("current_transaction")
        # The API now expects a list of full transaction objects, not a summary
        previous_transactions = data.get("previous_transactions")
        
        if not current_transaction or previous_transactions is None: # Allow an empty list
            return jsonify({"error": "Missing required keys: 'current_transaction' or 'previous_transactions'"}), 400

        # --- 2. Get Scam Score from ANN ---
        # NOTE: Your `preprocess_for_ann` function should also be updated to handle this new object
        processed_input = preprocess_for_ann(current_transaction)
        scam_score = ann_model.predict(processed_input)[0][0]
        scam_score = float(scam_score)
        
        # --- 3. MODIFIED: Prepare data and get Analysis from LLM ---
        # Calculate historical stats on the fly from the full transaction objects
        if previous_transactions:
            avg_value = np.mean([t.get('amount', 0) for t in previous_transactions])
        else:
            avg_value = 0
        
        # Reconstruct the historical_data object that the LLM function expects
        historical_data_for_llm = {
            "average_transaction_value": avg_value,
            "previous_transactions": previous_transactions
        }
        
        # The LLM analyzer expects a single object containing both current and historical data
        full_data_for_llm = {
            "current_transaction": current_transaction,
            "historical_data": historical_data_for_llm
        }
        llm_response = analyze_with_llm(full_data_for_llm, scam_score)

        # --- 4. Format and Return the Final Response ---
        final_response = {
            "scam_score": round(scam_score, 4),
            "llm_analysis": llm_response
        }
        return jsonify(final_response), 200

    except Exception as e:
        print(f"❌ An unexpected error occurred in /analyze endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
    
# --- This helper function is no longer used by the new /analyze_location but is kept for now ---
def geocode_location(location_name):
    """Helper function to convert a single location name to coordinates."""
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Geocoding error for '{location_name}': {e}")
    return None

@app.route('/analyze_location', methods=['POST'])
def analyze_location_endpoint():
    """
    MODIFIED: Analyzes geolocation using coordinates directly from the schema.
    This is much more efficient and reliable than geocoding addresses.
    """
    data = request.get_json()
    if not data or "current_geolocation" not in data or "previous_geolocations" not in data:
        return jsonify({"error": "Request must include 'current_geolocation' and 'previous_geolocations'"}), 400

    current_geo = data["current_geolocation"]
    previous_geos = data["previous_geolocations"]

    # --- Step 1: MODIFIED - Extract coordinates directly from objects ---
    try:
        current_coord = (current_geo['latitude'], current_geo['longitude'])
        # Filter out any malformed or empty geolocation objects
        previous_coords = [
            (g['latitude'], g['longitude']) for g in previous_geos 
            if g and 'latitude' in g and 'longitude' in g
        ]
    except (TypeError, KeyError) as e:
        return jsonify({"error": f"Invalid geolocation format provided: {e}"}), 400

    if not previous_coords:
        return jsonify({"error": "No valid previous geolocations were provided."}), 400

    # --- Step 2: Find the central point (centroid) of past locations ---
    kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(previous_coords)
    centroid = kmeans.cluster_centers_[0]

    # --- Step 3: Calculate distance from the new location to the center ---
    distance_km = geodesic(current_coord, centroid).kilometers

    # --- Step 4: Normalize the distance to a score between 0 and 1 ---
    anomaly_score = 1 - np.exp(-0.005 * distance_km)

    # --- Step 5: Return the analysis ---
    response = {
        "current_location_coordinates": current_coord,
        "historical_center_coordinates": list(centroid),
        "distance_from_center_km": round(distance_km, 2),
        "location_anomaly_score": round(anomaly_score, 4)
    }
    return jsonify(response), 200


if __name__ == '__main__':
    # I have kept the port at 5000 for consistency with your test scripts.
    app.run(debug=True, port=7000)

