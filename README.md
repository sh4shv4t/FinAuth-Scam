## Fraud Detection Backend
This backend service analyzes financial transactions to detect potential fraud. It uses a two-step process:

ANN-based Scoring: A pre-trained Artificial Neural Network (ANN) model first provides a "scam score" based on the current transaction's features.

LLM-based Analysis: The scam score, along with the current and historical transaction data, is then sent to a Large Language Model (LLM) for a final, human-readable analysis and a reason for flagging the transaction as fraudulent.

Project Structure
.
├── models/
│   └── fraud_detection_ann.h5  # Placeholder for your trained ANN model
├── main.py                     # Main Flask application
├── model_loader.py             # Utility to load the ANN model
├── llm_analyzer.py             # Module to interact with the LLM API
├── requirements.txt            # Python dependencies
└── README.md                   # This file

Setup
Clone the repository:

git clone <your-repo-url>
cd <your-repo-name>

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the dependencies:

pip install -r requirements.txt

Place your trained model:

Make sure your trained Keras/TensorFlow model is saved as fraud_detection_ann.h5 inside a models/ directory.

Set up your LLM API Key:

Create a .env file in the root of the project and add your LLM API key:

LLM_API_KEY="your_llm_api_key_here"

Running the Application
To start the Flask server, run:

python main.py

The server will start on http://127.0.0.1:5000.

API Endpoint
POST /analyze_transaction
This endpoint analyzes a new transaction for potential fraud.

Request Body (JSON):

{
  "current_transaction": {
    "amount": 2500.00,
    "transaction_time": "22:15",
    "location": "Online",
    "merchant_category": "Electronics"
  },
  "historical_data": {
    "previous_transactions": [
      {"amount": 50.00, "merchant_category": "Groceries"},
      {"amount": 120.00, "merchant_category": "Apparel"},
      {"amount": 30.00, "merchant_category": "Dining"},
      {"amount": 200.00, "merchant_category": "Travel"},
      {"amount": 75.00, "merchant_category": "Utilities"},
      {"amount": 90.00, "merchant_category": "Entertainment"},
      {"amount": 45.00, "merchant_category": "Groceries"},
      {"amount": 150.00, "merchant_category": "Apparel"},
      {"amount": 60.00, "merchant_category": "Dining"},
      {"amount": 180.00, "merchant_category": "Travel"}
    ],
    "average_transaction_value": 150.50,
    "average_bank_account_value": 5000.00,
    "average_transaction_time": "14:30"
  }
}

Success Response (200 OK):

{
  "ann_scam_score": 0.92,
  "is_fraudulent": true,
  "reason": "The transaction amount of $2500.00 is significantly higher than the user's average transaction value of $150.50. Additionally, the transaction is in the 'Electronics' category, which does not align with the user's recent spending habits in 'Groceries', 'Apparel', and 'Dining'."
}

Error Response (400 Bad Request):

{
  "error": "Invalid input: Missing 'current_transaction' in request."
}
