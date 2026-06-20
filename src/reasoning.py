


def _format_skill_list(skills, max_items=3):
    if not skills:
        return None
    shown = skills[:max_items]
    if len(skills) > max_items:
        return f"{', '.join(shown)}, +{len(skills) - max_items} more"
    return ", ".join(shown)


def generate_reasoning(features: dict, score_result: dict) -> str:
    title = features.get("current_title") or "Unknown title"
    yoe = features.get("years_of_experience", 0)
    must_haves = features.get("matched_must_haves", [])
    disqualifiers = score_result.get("disqualifiers", [])

    if "non_technical_title_no_real_signal" in disqualifiers:
        return (
            f"{title} with {yoe} yrs experience; no production AI/ML signal "
            f"in career history despite any skill keywords listed -- title and "
            f"role history don't match the role's core requirements."
        )

    if "consulting_only" in disqualifiers:
        return (
            f"{title} with {yoe} yrs, entirely at consulting/services firms; "
            f"no evidence of product-company ownership of an AI system."
        )

    if "research_only_no_production" in disqualifiers:
        return (
            f"{title} with {yoe} yrs, profile reads academic/research-focused; "
            f"no clear production deployment experience found."
        )

    if "honeypot_skill_duration_mismatch" in disqualifiers:
        return (
            f"{title} with {yoe} yrs; profile lists expert-level proficiency on "
            f"multiple skills with little to no recorded usage history -- "
            f"flagged as an inconsistent profile."
        )

    skill_text = _format_skill_list(must_haves)
    parts = [f"{title} with {yoe} yrs experience"]

    if skill_text:
        parts.append(f"hands-on with {skill_text}")
    else:
        parts.append("no directly matched must-have skills found")

    gaps = []
    if features.get("job_hopper_flag"):
        gaps.append("tenure pattern shows frequent short stints")
    if features.get("architecture_drift_flag"):
        gaps.append("recent role reads more managerial than hands-on")
    if not features.get("nice_to_have_skill_count"):
        gaps.append("no fine-tuning/MLOps extras noted")

    behavior_bits = []
    days_inactive = features.get("days_since_active")
    if days_inactive is not None:
        if days_inactive <= 14:
            behavior_bits.append("active on platform recently")
        elif days_inactive > 120:
            behavior_bits.append(f"inactive for {days_inactive} days")
    if features.get("open_to_work_flag"):
        behavior_bits.append("marked open to work")
    response_rate = features.get("recruiter_response_rate")
    if response_rate is not None and response_rate < 0.3:
        behavior_bits.append(f"low recruiter response rate ({response_rate:.0%})")

    sentence1 = ", ".join(parts) + "."

    sentence2_bits = []
    if gaps:
        sentence2_bits.append(gaps[0])
    if behavior_bits:
        sentence2_bits.append(behavior_bits[0])

    if sentence2_bits:
        sentence2 = "; ".join(sentence2_bits).capitalize() + "."
        return f"{sentence1} {sentence2}"

    return sentence1


if __name__ == "__main__":
    import sys, json
    from feature_extractor import extract_features
    from scorer import score_candidate

    path = sys.argv[1] if len(sys.argv) > 1 else "sample5.jsonl"
    with open(path, "r", encoding="utf-8-sig") as fh:
        rows = []
        for line in fh:
            if not line.strip():
                continue
            cand = json.loads(line)
            feats = extract_features(cand)
            result = score_candidate(feats)
            reasoning = generate_reasoning(feats, result)
            rows.append((result["score"], result["candidate_id"], reasoning))

    rows.sort(key=lambda x: -x[0])
    for score, cid, reasoning in rows:
        print(f"{score:6.2f}  {cid}")
        print(f"        {reasoning}")
        print()