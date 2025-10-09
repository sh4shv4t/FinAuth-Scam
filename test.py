import requests
import json

# The URL where your Flask application is running
# Note: Ensure this port matches the one in your main.py file (e.g., 7000 or 5000)
API_URL = "http://127.0.0.1:7000/analyze"

# --- Test Data Setup based on the new TransactionSchema ---

# A list of historical transactions for context
historical_transactions = [
    {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 50.10,
        "receiverName": "Grocery Store",
        "receiverAccount": "9876543210",
        "transactionType": "debit",
        "timestamp": "2025-10-08T15:00:00Z",
        "description": "Weekly groceries",
        "geolocation": {"latitude": 40.7128, "longitude": -74.0060} # New York
    },
    {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 22.00,
        "receiverName": "Metro Transit",
        "receiverAccount": "1122334455",
        "transactionType": "debit",
        "timestamp": "2025-10-07T08:30:00Z",
        "description": "Metro card top-up",
        "geolocation": {"latitude": 40.7306, "longitude": -73.9352} # New York
    },
    {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 1500.00,
        "receiverName": "Paycheck Deposit",
        "receiverAccount": "self",
        "transactionType": "credit",
        "timestamp": "2025-10-05T09:00:00Z",
        "description": "Monthly salary",
        "geolocation": None
    }
]

# --- Test Case 1: A Normal, Legitimate Transaction ---
# This payload sends a full current_transaction and a list of previous_transactions.
legitimate_payload = {
    "current_transaction": {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 45.50,
        "receiverAccount": "9876543210",
        "receiverName": "Grocery Store",
        "transactionType": "debit",
        "timestamp": "2025-10-09T14:30:00Z",
        "description": "More groceries",
        "geolocation": {"latitude": 40.7128, "longitude": -74.0060} # New York
    },
    "previous_transactions": historical_transactions
}

# --- Test Case 2: A Highly Suspicious Transaction ---
# An anomalous transaction: large debit, unknown receiver, unusual time, foreign location.
suspicious_payload = {
    "current_transaction": {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 8500.00,
        "receiverAccount": "5551239876",
        "receiverName": "Online Gaming Hub",
        "transactionType": "debit",
        "timestamp": "2025-10-09T03:15:00Z",
        "description": "Purchase of in-game currency",
        "geolocation": {"latitude": 4.7110, "longitude": -74.0721} # Bogota, CO
    },
    "previous_transactions": historical_transactions
}

# --- Test Case 3: An Invalid Payload ---
# This payload is missing the 'previous_transactions' key, which the server now requires.
invalid_payload = {
    "current_transaction": {
        "userId": "60d5f1f77b8c4b2a8c8b4567",
        "amount": 100.00,
        "transactionType": "debit",
        "timestamp": "2025-10-09T10:00:00Z"
    }
    # Missing "previous_transactions"
}

def test_transaction_endpoint(payload, description):
    """Sends a payload to the /analyze endpoint and prints the response."""
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
    test_transaction_endpoint(legitimate_payload, "Testing a LEGITIMATE Transaction")
    test_transaction_endpoint(suspicious_payload, "Testing a SUSPICIOUS Transaction")
    test_transaction_endpoint(invalid_payload, "Testing an INVALID Payload")

