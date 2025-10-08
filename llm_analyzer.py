import os
import requests
import logging
import json # Import the json library
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
    Sends transaction data and the ANN scam score to an LLM for analysis.
    This function signature matches what is expected by main.py.

    Args:
        full_data (dict): The complete request payload containing current and historical data.
        scam_score (float): The scam score from the ANN (0.0 to 1.0).

    Returns:
        A dictionary with the LLM's analysis, including a boolean flag and a reason.
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
    
    prompt = f"""
    Analyze the following financial transaction for potential fraud.

    Context:
    - A machine learning model has already assigned a fraud score of {scam_score:.2f} (out of 1.0).
    - Your role is to provide a final judgment and a clear, concise reason.

    User's Historical Data:
    - Average Transaction Value: ${historical_data.get('average_transaction_value')}
    - Average Bank Account Value: ${historical_data.get('average_bank_account_value')}
    - Normal Transaction Time: Around {historical_data.get('average_transaction_time')}
    - Recent Transaction Categories: {[t.get('merchant_category', 'N/A') for t in historical_data.get('previous_transactions', [])[:5]]}

    Current Transaction to Analyze:
    - Amount: ${transaction_data.get('amount')}
    - Time: {transaction_data.get('transaction_time')}
    - Location: {transaction_data.get('location')}
    - Merchant Category: {transaction_data.get('merchant_category')}

    Based on all the information above, please provide a JSON response with two keys:
    1. "is_fraudulent": A boolean (true if you suspect fraud, otherwise false).
    2. "reason": A brief, one-sentence explanation for your decision.
    
    Example response:
    {{
        "is_fraudulent": true,
        "reason": "The transaction amount is unusually high compared to the user's average and occurs at an odd time."
    }}
    """

    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
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

