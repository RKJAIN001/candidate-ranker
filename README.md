# Redrob Hackathon — AI Candidate Ranker

A rule-based, feature-driven candidate ranking system built for the Intelligent
Candidate Discovery & Ranking Challenge. Ranks 100,000 candidates against a
Senior AI Engineer job description and outputs the top 100, with grounded,
fact-based reasoning for each.

## Why rule-based instead of an LLM API call per candidate

The compute constraints (Section 3 of the submission spec) require the ranking
step to run in under 5 minutes, CPU-only, with no network access. That rules
out calling any hosted LLM (OpenAI, Anthropic, Gemini, etc.) during ranking.
Instead, this system extracts structured features from each candidate's raw
profile and scores them using an explicit, weighted rubric derived directly
from the job description's stated must-haves, nice-to-haves, and disqualifiers.

This also produces a system that is fully explainable and defensible — every
score and every line of reasoning traces back to a specific field in the
candidate's data, with no hallucination risk.

## Architecture
candidates.jsonl

│

▼

feature_extractor.py   → flat dict of ~30 clean signals per candidate

│

▼

scorer.py               → weighted composite score (0-100) + disqualifier flags

│

▼

reasoning.py             → 1-2 sentence fact-grounded explanation

│

▼

rank.py                  → orchestrates the above, outputs top 100 as CSV
## Scoring rubric

| Dimension | Weight | What it measures |
|---|---|---|
| Skills match | 30% | Must-have skill coverage (embeddings, vector DB, eval frameworks, Python), weighted by Redrob skill-assessment scores where available |
| Experience fit | 25% | Years of experience against the JD's ideal 5-9 year band, technical vs non-technical title signal |
| Credentials | 15% | Education tier, certifications |
| Career growth | 15% | Tenure health, job-hopping pattern, architecture-drift detection (senior title with no hands-on signal) |
| Platform signals | 15% | Recency of activity, open-to-work flag, recruiter response rate, notice period, GitHub activity |

### Why these specific penalty values

The penalties are multiplicative, not absolute zero, because none of these
patterns alone proves someone is a bad fit -- they're strong negative signals
that should dominate the score without making the system brittle to edge
cases.

- **Consulting-only (×0.35):** the mildest penalty of the four. Plenty of
  strong engineers spend years at a consulting firm before moving to
  product work, and the JD doesn't explicitly rule this out -- it's a
  caution flag, not proof of unfitness.
- **Research-only (×0.40):** similar logic. A strong research background
  with zero production signal is a real gap against this specific JD's
  "shipped to production" requirement, but research skills aren't worthless
  for this role, hence not as severe as the honeypot penalty.
- **Honeypot skill/duration mismatch (×0.10):** the harshest penalty,
  deliberately. This pattern signals the profile itself may not be
  trustworthy (claiming "expert" with near-zero usage), which is a
  different and more serious problem than someone simply being a weaker
  fit -- the JD and redrob_signals_doc both flag this as the trap the
  challenge is explicitly testing for.
- **Non-technical title with no real signal (×0.15):** the second-harshest
  penalty, since this is the dataset's central, explicitly named trap
  (sample_submission.csv's own #1-ranked candidate is this exact failure
  mode). It's not ×0.0 because we deliberately avoid hard zeroing any
  candidate -- a multiplicative floor keeps the system's behavior smooth
  and auditable rather than introducing a discontinuous cliff.

These values were set by reasoning about relative severity, not learned from
labeled data -- see Limitations & Future Work in the deck for how we'd
validate and tune them given more time.

### Hard disqualifiers (multiplicative penalty, not a hard zero)

- **Consulting-only career** (TCS/Infosys/Wipro/Accenture/etc. with no product company experience) → ×0.35
- **Research-only background** with no production deployment signal → ×0.40
- **Honeypot skill/duration mismatch** (e.g. "expert" proficiency with ≤3 months of usage, 2+ times) → ×0.10
- **Non-technical title with zero real AI/ML signal** in career history despite any skill keywords listed → ×0.15

This last rule is the direct counter to the challenge's central trap: a
candidate whose *title* doesn't match the role, regardless of what's listed
in their skills array, should not rank highly. This is exactly the failure
mode demonstrated by `sample_submission.csv`, where an HR Manager ranks #1
purely on keyword count.

## Verified results

On the full 100,000-candidate pool, on a standard laptop CPU:

- **Runtime: ~77-85 seconds** (well under the 5-minute limit)
- **Zero honeypots** in the top 100 (22 honeypot-pattern candidates exist in
  the full pool; none made the shortlist) — see `check_honeypots.py`
- **Zero non-technical/keyword-stuffer titles** in the top 100 — see `check_quality.py`
- Output passes `validate_submission.py` cleanly

## How to run

```bash
cd src
python rank.py --candidates ../data/candidates.jsonl --out ../output/submission.csv
```

No external dependencies — pure Python 3 standard library.

To verify the output:

```bash
cd ..
python validate_submission.py output/submission.csv
python check_quality.py
python check_honeypots.py
```

## Repository structure
candidate-ranker/

├── data/                    # candidates.jsonl goes here (not committed, see .gitignore)

├── src/

│   ├── feature_extractor.py

│   ├── scorer.py

│   ├── reasoning.py

│   └── rank.py

├── output/                  # submission.csv generated here

├── validate_submission.py   # official format validator

├── check_quality.py         # title/keyword-stuffer sanity check

├── check_honeypots.py       # honeypot leakage check

├── requirements.txt

└── README.md