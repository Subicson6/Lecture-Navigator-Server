import csv
import time
from statistics import quantiles
from pathlib import Path
import requests


API_URL = "http://localhost:8000/search_timestamps"


def load_gold(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["expected_t_start"] = float(r.get("expected_t_start", 0) or 0)
            rows.append(r)
    return rows


def mrr_at_k(ranks, k=10):
    rr = 0.0
    for r in ranks:
        if 1 <= r <= k:
            rr += 1.0 / r
    return rr / len(ranks) if ranks else 0.0


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "gold_set.csv"
    gold = load_gold(csv_path)

    ranks = []
    latencies = []

    for i, row in enumerate(gold, 1):
        payload = {
            "video_id": row["video_id"],
            "query": row["query"],
            "k": 10,
        }
        t0 = time.perf_counter()
        resp = requests.post(API_URL, json=payload, timeout=60)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        if resp.status_code != 200:
            print(f"[{i}] Request failed: {resp.status_code} {resp.text}")
            ranks.append(9999)
            continue

        data = resp.json()
        results = data.get("results", [])

        # Determine rank where expected_t_start hits any result window
        target = row["expected_t_start"]
        hit_rank = 9999
        for j, r in enumerate(results, 1):
            try:
                ts = float(r.get("t_start", 0.0))
                te = float(r.get("t_end", 0.0))
            except Exception:
                ts, te = 0.0, 0.0
            if ts <= target <= te:
                hit_rank = j
                break
        ranks.append(hit_rank)

    mrr = mrr_at_k(ranks, k=10)
    print(f"MRR@10: {mrr:.4f}")

    # p95 latency
    if latencies:
        p95 = quantiles(latencies, n=20)[18]  # approximate 95th percentile
        print(f"Latency p95: {p95:.1f} ms (n={len(latencies)})")


if __name__ == "__main__":
    main()
