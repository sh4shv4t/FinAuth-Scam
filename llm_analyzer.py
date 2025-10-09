import os
import requests
import logging
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key and URL from environment variables
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def analyze_with_llm(full_data, scam_score):
    """
    MODIFIED: Sends enriched transaction data based on the new schema to an LLM for analysis.
    """
    if not LLM_API_KEY:
        logger.error("LLM_API_KEY not found in environment variables.")
        return {
            "is_fraudulent": scam_score > 0.85, # Fallback logic
            "reason": "LLM analysis skipped: API key not configured. Flag based on high ANN score."
        }

    # Unpack the data for the prompt
    transaction_data = full_data.get('current_transaction', {})
    historical_data = full_data.get('historical_data', {})
    
    # --- MODIFIED: Create a richer summary of past transactions for the prompt ---
    previous_transaction_summary = [
        f"Type: {t.get('transactionType', 'N/A')}, Amount: ${t.get('amount', 0)}, To: {t.get('receiverName', 'N/A')}"
        for t in historical_data.get('previous_transactions', [])[:3]
    ]

    # --- MODIFIED: The prompt is now much more detailed using the new schema fields ---
    prompt = f"""
    Analyze the following financial transaction for potential fraud, acting as a senior risk analyst.

    Context:
    - A machine learning model has assigned an initial fraud score of {scam_score:.2f} (out of 1.0).
    - Your role is to provide a final judgment using the detailed context below.

    User's Historical Data:
    - Average Transaction Value: ${historical_data.get('average_transaction_value')}
    - A Sample of Recent Transactions: {previous_transaction_summary}

    Current Transaction to Analyze:
    - Transaction Type: {transaction_data.get('transactionType', 'N/A').upper()}
    - Amount: ${transaction_data.get('amount')}
    - Receiver Name: {transaction_data.get('receiverName', 'N/A')}
    - Receiver Account: {transaction_data.get('receiverAccount', 'N/A')}
    - Timestamp: {transaction_data.get('timestamp')}
    - Description: "{transaction_data.get('description', 'N/A')}"
    - Geolocation: {transaction_data.get('geolocation')}

    Based on all the information, especially comparing the current transaction's type, amount, and receiver to the user's history, provide a JSON response with two keys:
    1. "is_fraudulent": A boolean (true if you suspect fraud, otherwise false).
    2. "reason": A brief, one-sentence explanation for your decision.
    
    Example response:
    {{
        "is_fraudulent": true,
        "reason": "This is a large debit to an unknown receiver, which is highly unusual compared to the user's typical small, recurring grocery debits."
    }}
    """

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    full_url = f"{LLM_API_URL}?key={LLM_API_KEY}"

    try:
        response = requests.post(full_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        text_content = result['candidates'][0]['content']['parts'][0]['text']
        
        # Clean the response to make it valid JSON
        clean_json_str = text_content.strip().replace("```json", "").replace("```", "").strip()
        
        # Use json.loads() instead of eval() for safety
        return json.loads(clean_json_str)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling LLM API: {e}")
        return {
            "is_fraudulent": scam_score > 0.85,
            "reason": f"Could not get analysis from LLM due to an API error: {e}"
        }
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing LLM response: {e}")
        return {
            "is_fraudulent": scam_score > 0.85,
            "reason": "Could not parse the analysis from the LLM."
        }

