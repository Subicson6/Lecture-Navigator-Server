import requests
import pandas as pd
import numpy as np
import time
import csv
import os

# --- ⚙️ Configuration ---
# Your FastAPI server's base URL
API_BASE_URL = "http://127.0.0.1:8000"
# The endpoint you want to test (for full RAG path with LLM cost)
EVAL_ENDPOINT = f"{API_BASE_URL}/qa_video"
# The name of your evaluation dataset file
EVAL_DATASET_PATH = "evaluation.csv" # <<< Make sure this matches your filename
# The output file for the detailed report
REPORT_OUTPUT_PATH = "metrics_report.csv"
# The number of top results to request for calculating MRR@k
K_FOR_MRR = 10
# A result is considered a "hit" if its timestamp is within this many seconds of the expected time
MRR_TOLERANCE_SECONDS = 15

# --- 💲 LLM Cost Configuration (adjust for your model) ---
# Example prices for a model like Gemini 1.5 Flash
LLM_PRICE_PER_MILLION_INPUT_TOKENS = float(os.getenv("LLM_PRICE_INPUT", 0.35))
LLM_PRICE_PER_MILLION_OUTPUT_TOKENS = float(os.getenv("LLM_PRICE_OUTPUT", 0.70))

def calculate_reciprocal_rank(results, expected_start_time, tolerance_seconds):
    """Calculates the reciprocal rank for a single query."""
    for i, result in enumerate(results):
        # Check if the expected time is within the tolerance window of the result's start time
        if abs(result['t_start'] - expected_start_time) <= tolerance_seconds:
            return 1 / (i + 1)
    return 0.0  # Return 0 if not found in the top 'k' results

def run_evaluation():
    """Runs the full evaluation against the API and returns detailed results."""
    try:
        df = pd.read_csv(EVAL_DATASET_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Evaluation file not found at '{EVAL_DATASET_PATH}'. Please create it and add your test data.")
        return None, None, 0, 0

    all_results_data = []
    all_latencies = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    print(f"🚀 Starting evaluation of {len(df)} queries against '{EVAL_ENDPOINT}'...")
    print("-" * 60)

    for index, row in df.iterrows():
        query = row['query']
        video_id = row['video_id']
        expected_time = row['expected_start_time']

        payload = {"query": query, "video_id": video_id, "k": K_FOR_MRR}

        latency = 0.0
        reciprocal_rank = 0.0
        http_status = 0
        answer_snippet = ""
        prompt_tokens = 0
        completion_tokens = 0
        query_cost = 0.0

        try:
            start_time = time.perf_counter()
            response = requests.post(EVAL_ENDPOINT, json=payload, timeout=45) # 45-second timeout for the API call
            end_time = time.perf_counter()

            latency = end_time - start_time
            all_latencies.append(latency)
            http_status = response.status_code

            if response.status_code == 200:
                response_data = response.json()
                results_list = response_data.get("results", [])
                answer_snippet = response_data.get("answer", "")[:100].replace("\n", " ")
                prompt_tokens = response_data.get("prompt_tokens", 0)
                completion_tokens = response_data.get("completion_tokens", 0)

                reciprocal_rank = calculate_reciprocal_rank(results_list, expected_time, MRR_TOLERANCE_SECONDS)

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                query_cost = (prompt_tokens / 1_000_000 * LLM_PRICE_PER_MILLION_INPUT_TOKENS) + \
                             (completion_tokens / 1_000_000 * LLM_PRICE_PER_MILLION_OUTPUT_TOKENS)

            else:
                print(f"  ⚠️  API Error for query '{query[:30]}...': Status {http_status}")
                answer_snippet = response.text[:100]

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Network/Timeout Error for query '{query[:30]}...': {e}")
            http_status = 503 # Service Unavailable

        all_results_data.append({
            "query": query,
            "video_id": video_id,
            "latency_seconds": latency,
            "reciprocal_rank": reciprocal_rank,
            "http_status": http_status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_query_cost_usd": query_cost,
            "llm_answer_snippet": answer_snippet,
        })
        print(f"  Processed query {index + 1}/{len(df)} | Latency: {latency:.2f}s | RR: {reciprocal_rank:.2f} | Status: {http_status}")

    return all_results_data, all_latencies, total_prompt_tokens, total_completion_tokens

def generate_report(detailed_results, latencies, prompt_tokens, completion_tokens):
    """Calculates summary metrics and saves the detailed report to a CSV file."""
    if not detailed_results:
        print("\nNo results to generate a report from.")
        return

    # --- Save Detailed Report ---
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(REPORT_OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ Detailed report saved to '{REPORT_OUTPUT_PATH}'")

    # --- Calculate Aggregate Metrics ---
    mrr_at_k = df_detailed["reciprocal_rank"].mean()
    mean_latency = df_detailed["latency_seconds"].mean()
    p95_latency = np.percentile(latencies, 95) if latencies else 0.0

    # Error Budget Calculation
    total_requests = len(df_detailed)
    non_200_requests = len(df_detailed[df_detailed["http_status"] != 200])
    error_rate = (non_200_requests / total_requests) * 100 if total_requests > 0 else 0.0

    # Total Cost Calculation
    total_cost_usd = (prompt_tokens / 1_000_000 * LLM_PRICE_PER_MILLION_INPUT_TOKENS) + \
                     (completion_tokens / 1_000_000 * LLM_PRICE_PER_MILLION_OUTPUT_TOKENS)

    # --- Print Summary to Console ---
    print("\n" + "="*25 + " 📊 METRICS SUMMARY " + "="*25)
    print(f"|  ACCURACY (MRR@{K_FOR_MRR}):\t\t{mrr_at_k:.4f}")
    print(f"|  LATENCY (Mean):\t\t{mean_latency:.4f} seconds")
    print(f"|  LATENCY (p95):\t\t{p95_latency:.4f} seconds")
    print(f"|  RELIABILITY (Error Rate):\t{error_rate:.2f}%")
    print(f"|  COST (Total Estimated):\t${total_cost_usd:.6f} USD for {total_requests} queries")
    print(f"|    - Total Prompt Tokens:\t{prompt_tokens}")
    print(f"|    - Total Completion Tokens:\t{completion_tokens}")
    print("="*70)
    print("\n--- ✅ Compliance Check ---")
    print(f"p95 latency ≤ 2.5s: {'PASS' if p95_latency <= 2.5 else 'FAIL'}")
    print(f"Error Rate < 0.5%:  {'PASS' if error_rate < 0.5 else 'FAIL'}")
    print("-" * 28 + "\n")


if __name__ == "__main__":
    results, latencies, p_tokens, c_tokens = run_evaluation()
    generate_report(results, latencies, p_tokens, c_tokens)