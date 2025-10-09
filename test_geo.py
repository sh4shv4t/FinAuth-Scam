import requests
import json

# The URL should match the port in your main.py file (e.g., 7000)
API_URL = "http://127.0.0.1:7000/analyze_location"

# --- Test Case 1: Low Anomaly Location ---
# MODIFIED: Payload now sends coordinate objects instead of strings.
# Pune is geographically close to the historical cluster in India.
low_anomaly_payload = {
    "current_geolocation": {"latitude": 18.5204, "longitude": 73.8567}, # Pune, India
    "previous_geolocations": [
        {"latitude": 18.5204, "longitude": 73.8567}, # Pune, India
        {"latitude": 28.7041, "longitude": 77.1025}, # Delhi, India
        {"latitude": 19.0760, "longitude": 72.8777}, # Mumbai, India
        {"latitude": 18.5204, "longitude": 73.8567}  # Pune, India
    ]
}

# --- Test Case 2: High Anomaly Location ---
# MODIFIED: Payload now sends coordinate objects.
# Moscow is very far from the historical cluster.
high_anomaly_payload = {
    "current_geolocation": {"latitude": 55.7558, "longitude": 37.6173}, # Moscow, Russia
    "previous_geolocations": [
        {"latitude": 26.9124, "longitude": 75.7873}, # Jaipur, India
        {"latitude": 28.7041, "longitude": 77.1025}, # Delhi, India
        {"latitude": 19.0760, "longitude": 72.8777}, # Mumbai, India
        {"latitude": 12.9716, "longitude": 77.5946}  # Bangalore, India
    ]
}

# --- Test Case 3: Invalid Payload ---
# MODIFIED: Payload is missing the 'previous_geolocations' key.
invalid_payload = {
    "current_geolocation": {"latitude": 51.5074, "longitude": -0.1278} # London, UK
    # Missing "previous_geolocations"
}


def test_location_endpoint(payload, description):
    """
    Sends a location payload to the /analyze_location endpoint and prints the response.
    """
    print(f"\n{'='*20} {description.upper()} {'='*20}")
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        response.raise_for_status()

        print(f"✅ SUCCESS: Server responded with status code {response.status_code}")
        print("Server Response:")
        print(json.dumps(response.json(), indent=4))

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP ERROR: Server responded with status code {http_err.response.status_code}")
        print("Server Error Message:")
        try:
            print(http_err.response.json())
        except json.JSONDecodeError:
            print(http_err.response.text)
    except requests.exceptions.RequestException as err:
        print(f"❌ CONNECTION ERROR: Could not connect to the server at {API_URL}.")
        print("Please make sure your Flask server is running.")
        print(f"Error details: {err}")
    print('=' * (42 + len(description)))


if __name__ == "__main__":
    # Run all the test cases
    test_location_endpoint(low_anomaly_payload, "Testing a LOW Anomaly Location")
    test_location_endpoint(high_anomaly_payload, "Testing a HIGH Anomaly Location")
    test_location_endpoint(invalid_payload, "Testing an INVALID Payload")

