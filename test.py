import requests
import json

# --- FIXED THE TYPO IN THE URL HERE ---
# The URL where your Flask application is running
API_URL = "http://127.0.0.1:5000/analyze"

# --- Test Case 1: A Normal, Legitimate Transaction ---
# This payload represents a typical, low-risk transaction.
legitimate_payload = {
    "current_transaction": {
        "amount": 45.50,
        "transaction_time": "14:30",
        "merchant_category": "Groceries",
        "location": "New York, NY"
    },
    "historical_data": {
        "previous_transactions": [
            {"amount": 50.10, "merchant_category": "Groceries"},
            {"amount": 22.00, "merchant_category": "Transport"},
            {"amount": 15.75, "merchant_category": "Cafe"},
            {"amount": 120.00, "merchant_category": "Shopping"},
            {"amount": 48.00, "merchant_category": "Groceries"}
        ],
        "average_transaction_value": 51.17,
        "average_bank_account_value": 4500.00,
        "average_transaction_time": "15:00"
    }
}

# --- Test Case 2: A Highly Suspicious Transaction ---
# This payload represents an anomaly.
suspicious_payload = {
    "current_transaction": {
        "amount": 8500.00,
        "transaction_time": "03:15",
        "merchant_category": "Online Gaming",
        "location": "Bogota, CO"
    },
    "historical_data": {
         "previous_transactions": [
            {"amount": 50.10, "merchant_category": "Groceries"},
            {"amount": 22.00, "merchant_category": "Transport"},
            {"amount": 15.75, "merchant_category": "Cafe"},
            {"amount": 120.00, "merchant_category": "Shopping"},
            {"amount": 48.00, "merchant_category": "Groceries"}
        ],
        "average_transaction_value": 51.17,
        "average_bank_account_value": 4500.00,
        "average_transaction_time": "15:00"
    }
}

# --- Test Case 3: An Invalid Payload ---
# This payload is missing the required 'historical_data' key.
invalid_payload = {
    "current_transaction": {
        "amount": 100.00,
        "transaction_time": "10:00",
        "merchant_category": "Utilities",
        "location": "London, UK"
    }
    # Missing "historical_data"
}


def test_transaction_endpoint(payload, description):
    """
    Sends a payload to the /analyze endpoint and prints the response.
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
        # Use .text for non-JSON error responses (like 404 HTML pages)
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
    test_transaction_endpoint(legitimate_payload, "Testing a LEGITIMATE Transaction")
    test_transaction_endpoint(suspicious_payload, "Testing a SUSPICIOUS Transaction")
    test_transaction_endpoint(invalid_payload, "Testing an INVALID Payload")

