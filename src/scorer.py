from feature_extractor import MUST_HAVE_SKILLS
WEIGHTS = {
    "skills_match": 0.30,
    "experience_fit": 0.25,
    "credentials": 0.15,
    "career_growth": 0.15,
    "platform_signals": 0.15,
}



IDEAL_YOE_MIN = 5
IDEAL_YOE_MAX = 9
TOTAL_MUST_HAVES = len(MUST_HAVE_SKILLS)


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _skills_subscore(f: dict) -> float:
    must_have_ratio = _clamp(f["must_have_skill_count"] / max(TOTAL_MUST_HAVES, 1) * 3.0)
    nice_to_have_bonus = _clamp(f["nice_to_have_skill_count"] / 5.0) * 0.15

    assessment = f.get("avg_skill_assessment_score")
    assessment_factor = 1.0
    if assessment is not None:
        assessment_factor = 0.7 + 0.3 * (assessment / 100.0)

    raw = (must_have_ratio * 0.85 + nice_to_have_bonus) * assessment_factor
    return _clamp(raw)


def _experience_subscore(f: dict) -> float:
    yoe = f["years_of_experience"]

    if yoe < IDEAL_YOE_MIN:
        yoe_score = _clamp(yoe / IDEAL_YOE_MIN)
    elif yoe <= IDEAL_YOE_MAX:
        yoe_score = 1.0
    else:
        overflow = yoe - IDEAL_YOE_MAX
        yoe_score = _clamp(1.0 - overflow * 0.04)

    title_bonus = 0.15 if f["title_is_technical"] else 0.0
    title_penalty = 0.35 if f["title_is_non_technical"] else 0.0

    raw = yoe_score * 0.7 + title_bonus - title_penalty
    return _clamp(raw)


def _credentials_subscore(f: dict) -> float:
    tier_score = f["best_education_tier_score"] / 4.0
    cert_bonus = _clamp(f["num_certifications"] / 3.0) * 0.2
    return _clamp(tier_score * 0.8 + cert_bonus)


def _career_growth_subscore(f: dict) -> float:
    score = 1.0

    if f["job_hopper_flag"]:
        score -= 0.45
    if f["architecture_drift_flag"]:
        score -= 0.35

    avg_tenure = f["avg_tenure_months"]
    if avg_tenure < 10:
        score -= 0.15
    elif avg_tenure > 60:
        score -= 0.05

    return _clamp(score)


def _platform_signals_subscore(f: dict) -> float:
    score = 0.0
    weight_sum = 0.0

    days_inactive = f.get("days_since_active")
    if days_inactive is not None:
        recency_score = _clamp(1.0 - days_inactive / 180.0)
        score += recency_score * 0.35
        weight_sum += 0.35

    if f["open_to_work_flag"]:
        score += 1.0 * 0.20
    weight_sum += 0.20

    response_rate = f.get("recruiter_response_rate", 0) or 0
    score += response_rate * 0.20
    weight_sum += 0.20

    notice = f.get("notice_period_days", 30) or 0
    notice_score = _clamp(1.0 - notice / 90.0)
    score += notice_score * 0.15
    weight_sum += 0.15

    gh = f.get("github_activity_score")
    if gh is not None:
        score += (gh / 100.0) * 0.10
        weight_sum += 0.10

    return _clamp(score / weight_sum) if weight_sum > 0 else 0.5


def score_candidate(f: dict) -> dict:
    sub = {
        "skills_match": _skills_subscore(f),
        "experience_fit": _experience_subscore(f),
        "credentials": _credentials_subscore(f),
        "career_growth": _career_growth_subscore(f),
        "platform_signals": _platform_signals_subscore(f),
    }

    composite = sum(sub[k] * WEIGHTS[k] for k in WEIGHTS)

    disqualifiers = []

    if f["consulting_only_flag"]:
        composite *= 0.35
        disqualifiers.append("consulting_only")

    if f["research_only_flag"]:
        composite *= 0.40
        disqualifiers.append("research_only_no_production")

    if f["honeypot_skill_mismatch_count"] >= 2:
        composite *= 0.10
        disqualifiers.append("honeypot_skill_duration_mismatch")

    if f["title_is_non_technical"] and f["must_have_skill_count"] == 0:
        composite *= 0.15
        disqualifiers.append("non_technical_title_no_real_signal")

    composite_pct = round(_clamp(composite) * 100, 2)

    return {
        "candidate_id": f["candidate_id"],
        "score": composite_pct,
        "subscores": {k: round(v, 3) for k, v in sub.items()},
        "disqualifiers": disqualifiers,
    }


if __name__ == "__main__":
    import sys, json
    from feature_extractor import extract_features

    path = sys.argv[1] if len(sys.argv) > 1 else "sample5.jsonl"
    with open(path, "r", encoding="utf-8-sig") as fh:
        results = []
        for line in fh:
            if not line.strip():
                continue
            cand = json.loads(line)
            feats = extract_features(cand)
            result = score_candidate(feats)
            results.append((result, feats["current_title"]))

    results.sort(key=lambda x: -x[0]["score"])
    for r, title in results:
        flags = f"  FLAGS: {r['disqualifiers']}" if r["disqualifiers"] else ""
        print(f"{r['score']:6.2f}  {r['candidate_id']}  ({title}){flags}")