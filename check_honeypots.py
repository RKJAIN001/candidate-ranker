import json
import csv

honeypot_ids = set()

with open("data/candidates.jsonl", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        skills = d.get("skills", [])

        # honeypot pattern: expert/advanced proficiency claimed with
        # near-zero months of actual usage, happening 2+ times
        bad_skill_hits = sum(
            1 for s in skills
            if s.get("proficiency", "").lower() in ("expert", "advanced")
            and (s.get("duration_months", 0) or 0) <= 3
        )
        if bad_skill_hits >= 2:
            honeypot_ids.add(d["candidate_id"])

print(f"Total honeypot-pattern candidates found in full 100K pool: {len(honeypot_ids)}")

with open("output/submission.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    top100_ids = {r["candidate_id"] for r in reader}

overlap = honeypot_ids & top100_ids
print(f"Honeypots that leaked into our top 100: {len(overlap)}")
print(f"Honeypot rate in our shortlist: {len(overlap) / 100 * 100:.1f}%  (must stay <= 10% to avoid disqualification)")

if overlap:
    print("Leaked IDs:", overlap)
else:
    print("Clean -- no honeypots in shortlist.")