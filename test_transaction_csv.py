import asyncio
import aiohttp
import json
import pandas as pd
from colorama import init, Fore, Style
import time

# Initialize colorama
init(autoreset=True)

API_URL = "http://127.0.0.1:7000/analyze"
CSV_FILE_PATH = "test_transactions.csv"

# --- MODIFIED: Further reduced concurrency and added a delay to respect API limits ---
CONCURRENT_REQUESTS = 2  # A safer value for strict free-tier APIs
REQUEST_DELAY_SECONDS = 0.3 # Adds a small pause between requests

async def fetch_analysis(session, semaphore, payload, transaction_info, index):
    """Fetches analysis for a single transaction, respecting the semaphore and delay."""
    async with semaphore:
        try:
            headers = {'Content-Type': 'application/json'}
            async with session.post(API_URL, data=json.dumps(payload), headers=headers, timeout=30) as response:
                
                # Add a small delay after each request is sent
                await asyncio.sleep(REQUEST_DELAY_SECONDS)

                if response.status == 200:
                    result = await response.json()
                    return (index, transaction_info, result) # Return the full result for ordered printing
                else:
                    error_json = await response.json()
                    return (index, transaction_info, {"error": f"HTTP {response.status}", "message": error_json})

        except asyncio.TimeoutError:
            return (index, transaction_info, {"error": "Timeout", "message": "The request took too long."})
        except aiohttp.ClientConnectorError:
            return (index, transaction_info, {"error": "Connection Failed", "message": f"Could not connect to {API_URL}."})

async def run_batch_analysis():
    """Reads transactions and sends them to the API concurrently with rate limiting."""
    start_time = time.time()
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ Successfully loaded {len(df)} transactions from '{CSV_FILE_PATH}'.\n")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{CSV_FILE_PATH}' was not found.")
        return

    historical_transactions = [
        {"amount": 50.10, "transactionType": "debit", "receiverName": "GroceryMart"},
        {"amount": 22.00, "transactionType": "debit", "receiverName": "Metro Transit"},
        {"amount": 2500.00, "transactionType": "credit", "receiverName": "Salary Deposit"}
    ]

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for index, row in df.iterrows():
            current_transaction = {
                "userId": row["userId"], "amount": row["amount"], "receiverAccount": row["receiverAccount"],
                "receiverName": row["receiverName"], "transactionType": row["transactionType"],
                "timestamp": row["timestamp"], "description": row["description"],
                "geolocation": {"latitude": row["geolocation_lat"], "longitude": row["geolocation_lon"]}
            }
            payload = {"current_transaction": current_transaction, "previous_transactions": historical_transactions}
            
            task = fetch_analysis(session, semaphore, payload, row['description'], index)
            tasks.append(task)
        
        # --- MODIFIED: Gather all results before printing ---
        all_results = await asyncio.gather(*tasks)

    # --- MODIFIED: Sort results and print them in order ---
    all_results.sort(key=lambda x: x[0]) # Sort by the original index

    print("\n--- Batch Analysis Complete ---\n")
    for index, transaction_info, result in all_results:
        print(f"{Style.BRIGHT}--- Result for Transaction #{index + 1} ({transaction_info}) ---")
        if "error" in result:
            print(Fore.YELLOW + f"⚠️  ERROR: {result['error']}")
            print(Fore.YELLOW + f"    Details: {result['message']}")
            if result['error'] == "Connection Failed":
                break
        else:
            llm_analysis = result.get("llm_analysis", {})
            is_fraudulent = llm_analysis.get("is_fraudulent", False)
            reason = llm_analysis.get("reason", "No reason provided.")
            scam_score = result.get("scam_score", 0)

            if is_fraudulent:
                print(Fore.RED + f"🔴 FLAGGED AS FRAUDULENT (Score: {scam_score:.2f})")
                print(Fore.RED + f"   Reason: {reason}")
            else:
                print(Fore.GREEN + f"🟢 Marked as Legitimate (Score: {scam_score:.2f})")
                print(Fore.GREEN + f"   Reason: {reason}")
        print("-" * 50)


    end_time = time.time()
    print(f"\nTime taken to execute the script: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(run_batch_analysis())

