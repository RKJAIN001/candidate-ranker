"""
feature_extractor.py
---------------------
Turns one raw candidate JSON record into a flat dict of clean, numeric/boolean
features that the scorer can consume. No LLM calls here -- pure parsing logic.

Run standalone for a quick sanity check:
    python feature_extractor.py
"""

import json
import re
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Reference vocab pulled from the JD -- edit these lists if the JD changes
# ---------------------------------------------------------------------------

MUST_HAVE_SKILLS = {
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "openai embeddings", "embeddings",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector search", "hybrid search",
    "python",
    "ndcg", "mrr", "map", "a/b testing", "ab testing", "offline evaluation",
    "learning to rank", "ranking evaluation",
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "lightgbm", "learning-to-rank", "neural ranking",
    "kubernetes", "docker", "distributed systems", "spark", "kafka",
    "open source", "oss",
}

# Title substrings that signal a NON-technical role wearing AI keywords
# (this is the exact trap called out in job_description.docx)
NON_TECHNICAL_TITLE_PATTERNS = re.compile(
    r"\b(hr|human resources|recruiter|recruiting|marketing|sales|"
    r"content writer|copywriter|graphic designer|accountant|"
    r"customer (support|success)|operations manager|business analyst|"
    r"product marketing)\b",
    re.IGNORECASE,
)

TECHNICAL_TITLE_PATTERNS = re.compile(
    r"\b(software|backend|frontend|full.?stack|machine learning|ml |"
    r"\bai\b|data scien|data engineer|applied ml|research engineer|"
    r"developer|ml engineer)\b",
    re.IGNORECASE,
)


NON_AI_ENGINEER_PATTERNS = re.compile(
    r"\b(civil|mechanical|electrical|chemical|structural|industrial|"
    r"qa engineer|quality assurance)\b",
    re.IGNORECASE,
)

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy", "hcl", "tech mahindra",
}

RESEARCH_ONLY_PATTERNS = re.compile(
    r"\b(phd|postdoc|research (fellow|associate|scientist)|academia|university)\b",
    re.IGNORECASE,
)


def _safe_lower(s):
    return (s or "").strip().lower()


def extract_features(candidate: dict) -> dict:
    """Take one raw candidate record, return a flat feature dict."""

    cid = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    certs = candidate.get("certifications", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    features = {"candidate_id": cid}

    # ---------------- Profile-level basics ----------------
    features["current_title"] = profile.get("current_title", "")
    features["years_of_experience"] = profile.get("years_of_experience", 0) or 0
    features["location"] = profile.get("location", "")
    features["current_industry"] = profile.get("current_industry", "")

    title_text = _safe_lower(profile.get("current_title", ""))
    headline_text = _safe_lower(profile.get("headline", ""))
    summary_text = _safe_lower(profile.get("summary", ""))

    features["title_is_non_technical"] = bool(
        NON_TECHNICAL_TITLE_PATTERNS.search(title_text)
    )
    features["title_is_technical"] = bool(
        TECHNICAL_TITLE_PATTERNS.search(title_text)
    ) and not bool(NON_AI_ENGINEER_PATTERNS.search(title_text))

    # ---------------- Career history aggregates ----------------
    all_descriptions = " ".join(
        _safe_lower(role.get("description", "")) for role in career
    )
    all_titles = " ".join(_safe_lower(role.get("title", "")) for role in career)
    all_companies = [_safe_lower(role.get("company", "")) for role in career]

    features["career_text_blob"] = f"{headline_text} {summary_text} {all_titles} {all_descriptions}"

    durations = [role.get("duration_months", 0) or 0 for role in career]
    features["num_roles"] = len(career)
    features["avg_tenure_months"] = round(sum(durations) / len(durations), 1) if durations else 0
    features["total_career_months"] = sum(durations)

    # consulting-only flag: every company matches a consulting firm name
    features["consulting_only_flag"] = bool(career) and all(
        any(firm in comp for firm in CONSULTING_FIRMS) for comp in all_companies
    )

    # research-only flag: academia signals + barely any industry roles
    features["research_only_flag"] = bool(
        RESEARCH_ONLY_PATTERNS.search(summary_text)
        or RESEARCH_ONLY_PATTERNS.search(headline_text)
    ) and len(career) <= 1

    # job-hopping flag: 3+ roles all under 20 months average tenure
    features["job_hopper_flag"] = (
        len(career) >= 3 and features["avg_tenure_months"] < 20
    )

    # architecture-drift flag: senior title but no hands-on language in current role
    hands_on_terms = ("implement", "built", "wrote", "coded", "shipped", "deployed", "developed")
    current_role = next((r for r in career if r.get("is_current")), None)
    if current_role:
        cur_desc = _safe_lower(current_role.get("description", ""))
        cur_title = _safe_lower(current_role.get("title", ""))
        is_senior_title = any(t in cur_title for t in ("senior", "staff", "principal", "lead", "head", "director"))
        has_hands_on = any(term in cur_desc for term in hands_on_terms)
        features["architecture_drift_flag"] = is_senior_title and not has_hands_on
    else:
        features["architecture_drift_flag"] = False

    # ---------------- Skills ----------------
    skill_names_lower = {_safe_lower(s.get("name", "")) for s in skills}
    matched_must_haves = {
        s for s in MUST_HAVE_SKILLS
        if s in skill_names_lower or s in features["career_text_blob"]
    }
    matched_nice_to_haves = {
        s for s in NICE_TO_HAVE_SKILLS
        if s in skill_names_lower or s in features["career_text_blob"]
    }
    features["must_have_skill_count"] = len(matched_must_haves)
    features["nice_to_have_skill_count"] = len(matched_nice_to_haves)
    features["matched_must_haves"] = sorted(matched_must_haves)
    features["matched_nice_to_haves"] = sorted(matched_nice_to_haves)

    # honeypot signal: skill claimed at "expert"/"advanced" but near-zero duration
    honeypot_skill_hits = 0
    for s in skills:
        prof = _safe_lower(s.get("proficiency", ""))
        dur = s.get("duration_months", 0) or 0
        if prof in ("expert", "advanced") and dur <= 3:
            honeypot_skill_hits += 1
    features["honeypot_skill_mismatch_count"] = honeypot_skill_hits

    features["total_endorsements"] = sum(s.get("endorsements", 0) or 0 for s in skills)

    # ---------------- Education ----------------
    tiers = [e.get("tier", "unknown") for e in education]
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    features["best_education_tier_score"] = max((tier_rank.get(t, 0) for t in tiers), default=0)
    features["num_certifications"] = len(certs)

    # ---------------- Behavioral / platform signals ----------------
    features["profile_completeness_score"] = signals.get("profile_completeness_score", 0) or 0
    features["open_to_work_flag"] = bool(signals.get("open_to_work_flag", False))
    features["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0) or 0
    features["notice_period_days"] = signals.get("notice_period_days", 0) or 0
    features["willing_to_relocate"] = bool(signals.get("willing_to_relocate", False))
    features["interview_completion_rate"] = signals.get("interview_completion_rate", 0) or 0

    gh = signals.get("github_activity_score", -1)
    features["github_activity_score"] = None if gh == -1 else gh

    offer_rate = signals.get("offer_acceptance_rate", -1)
    features["offer_acceptance_rate"] = None if offer_rate == -1 else offer_rate

    last_active = signals.get("last_active_date")
    features["last_active_date"] = last_active
    if last_active:
        try:
            la = datetime.strptime(last_active, "%Y-%m-%d").date()
            features["days_since_active"] = (date.today() - la).days
        except ValueError:
            features["days_since_active"] = None
    else:
        features["days_since_active"] = None

    skill_assessments = signals.get("skill_assessment_scores", {}) or {}
    features["avg_skill_assessment_score"] = (
        round(sum(skill_assessments.values()) / len(skill_assessments), 1)
        if skill_assessments else None
    )

    return features


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample5.jsonl"
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            cand = json.loads(line)
            feats = extract_features(cand)
            print(f"--- {feats['candidate_id']} ---")
            print(f"  title: {feats['current_title']!r}  non_technical={feats['title_is_non_technical']}  technical={feats['title_is_technical']}")
            print(f"  YOE: {feats['years_of_experience']}  consulting_only={feats['consulting_only_flag']}  job_hopper={feats['job_hopper_flag']}")
            print(f"  must_haves({feats['must_have_skill_count']}): {feats['matched_must_haves']}")
            print(f"  honeypot_skill_mismatch_count: {feats['honeypot_skill_mismatch_count']}")
            print(f"  github_activity_score: {feats['github_activity_score']}  days_since_active: {feats['days_since_active']}")
            print()