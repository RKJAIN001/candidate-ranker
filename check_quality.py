import csv

bad_title_words = ["hr manager", "marketing manager", "sales", "accountant",
                    "customer support", "operations manager", "recruiter",
                    "content writer", "graphic designer"]

with open("output/submission.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")

flagged = []
for r in rows:
    reasoning_lower = r["reasoning"].lower()
    if any(b in reasoning_lower[:40] for b in bad_title_words):
        flagged.append(r)

print(f"Rows with suspicious non-technical titles: {len(flagged)}")
for r in flagged:
    print(f"  rank={r['rank']}  {r['candidate_id']}  {r['reasoning'][:80]}")

print()
print("--- Bottom 10 (weakest in our shortlist) ---")
for r in rows[-10:]:
    print(f"  rank={r['rank']}  score={r['score']}  {r['reasoning'][:80]}")