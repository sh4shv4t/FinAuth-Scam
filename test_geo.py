import requests
import json

# The URL for your new location analysis endpoint
API_URL = "http://127.0.0.1:5000/analyze_location"

# --- Test Case 1: Low Anomaly Location ---
# The current location (Pune) is geographically close to the historical cluster in India.
# We expect a low anomaly score.
low_anomaly_payload = {
    "current_location": "Pune, India",
    "previous_locations": [
        "Jaipur, India",
        "Delhi, India",
        "Mumbai, India",
        "Bangalore, India"
    ]
}

# --- Test Case 2: High Anomaly Location ---
# The current location (Moscow) is very far from the historical cluster.
# We expect a high anomaly score (close to 1.0).
high_anomaly_payload = {
    "current_location": "Moscow, Russia",
    "previous_locations": [
        "Jaipur, India",
        "Delhi, India",
        "Mumbai, India",
        "Bangalore, India"
    ]
}

# --- Test Case 3: Invalid Payload ---
# This payload is missing the required 'previous_locations' key.
# We expect the server to catch this and return a 400 Bad Request error.
invalid_payload = {
    "current_location": "London, UK"
    # Missing "previous_locations"
}


def test_location_endpoint(payload, description):
    """
    Sends a location payload to the /analyze_location endpoint and prints the response.

    Args:
        payload (dict): The JSON data to send in the request body.
        description (str): A string describing the test case.
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
            print(http_err.response.text) # Fallback for non-JSON error pages
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