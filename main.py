import os
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- New Imports for Location Analysis ---
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.cluster import KMeans # <-- FIX: Added the missing import for KMeans

# Import your existing custom modules
from model_loader import load_model, preprocess_for_ann
from llm_analyzer import analyze_with_llm


# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# --- Initialize Geocoder for Location Analysis ---
# Using a descriptive user_agent is good practice
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
    
# --- NEW ENDPOINT FOR LOCATION ANALYSIS ---
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
    Analyzes a new transaction's location against a history of past locations.
    """
    data = request.get_json()
    if not data or "current_location" not in data or "previous_locations" not in data:
        return jsonify({"error": "Request must include 'current_location' and 'previous_locations'"}), 400

    current_loc_str = data["current_location"]
    previous_locs_str = data["previous_locations"]

    # --- Step 1: Geocode all location strings to coordinates ---
    previous_coords = [coord for coord in (geocode_location(loc) for loc in previous_locs_str) if coord]
    current_coord = geocode_location(current_loc_str)

    if not current_coord or not previous_coords:
        return jsonify({"error": "Could not geocode one or more locations."}), 400

    # --- Step 2: Find the central point (centroid) of past locations ---
    # We use KMeans with 1 cluster to find the mathematical center.
    kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto')
    kmeans.fit(previous_coords)
    centroid = kmeans.cluster_centers_[0]

    # --- Step 3: Calculate distance from the new location to the center ---
    distance_km = geodesic(current_coord, centroid).kilometers

    # --- Step 4: Normalize the distance to a score between 0 and 1 ---
    # This simple exponential function maps distance to a 0-1 score.
    # A small distance gives a score near 0; a large distance approaches 1.
    # The '0.005' factor can be tuned to change sensitivity.
    anomaly_score = 1 - np.exp(-0.005 * distance_km)

    # --- Step 5: Return the analysis ---
    response = {
        "current_location": {
            "name": current_loc_str,
            "coordinates": current_coord
        },
        "historical_center": {
            "coordinates": list(centroid)
        },
        "distance_from_center_km": round(distance_km, 2),
        "location_anomaly_score": round(anomaly_score, 4)
    }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)

