

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from feature_extractor import extract_features
from scorer import score_candidate
from reasoning import generate_reasoning

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
TOP_N = 100


def load_and_score(candidates_path: str):
    results = []
    skipped = 0

    with open(candidates_path, "r", encoding="utf-8-sig") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            try:
                features = extract_features(candidate)
                result = score_candidate(features)
                reasoning = generate_reasoning(features, result)
            except Exception as e:
               
                cid = candidate.get("candidate_id", f"line {line_num}")
                print(f"[error] {cid}: {type(e).__name__}: {e}", file=sys.stderr)
                skipped += 1
                continue

            results.append((result["score"], result["candidate_id"], reasoning))

    if skipped:
        print(f"[warn] skipped {skipped} malformed/unparseable records")

    return results


def select_top_n(results, n=TOP_N):
    results_sorted = sorted(results, key=lambda r: (-r[0], r[1]))
    top = results_sorted[:n]

    ranked_rows = []
    for i, (score, cid, reasoning) in enumerate(top, start=1):
        ranked_rows.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(score / 100.0, 4),
            "reasoning": reasoning,
        })
    return ranked_rows


def write_csv(rows, out_path: str):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write the output CSV")
    args = parser.parse_args()

    t0 = time.time()

    print(f"[1/3] Loading and scoring candidates from {args.candidates} ...")
    results = load_and_score(args.candidates)
    print(f"      scored {len(results)} candidates in {time.time() - t0:.1f}s")

    print(f"[2/3] Selecting top {TOP_N} and assigning ranks ...")
    ranked_rows = select_top_n(results, TOP_N)

    print(f"[3/3] Writing submission CSV to {args.out} ...")
    write_csv(ranked_rows, args.out)

    elapsed = time.time() - t0
    print(f"Done. Total runtime: {elapsed:.1f}s")
    print("Top 5 preview:")
    for row in ranked_rows[:5]:
        print(f"  rank={row['rank']:3d}  score={row['score']:.4f}  {row['candidate_id']}  {row['reasoning'][:70]}")


if __name__ == "__main__":
    main()