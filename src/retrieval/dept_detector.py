"""Automatic department detection from query keywords.

Uses simple keyword matching (no LLM call) to infer which department's
documents are most relevant to a query. Returns None when ambiguous.
"""
from __future__ import annotations

# Department -> keyword patterns (case-insensitive).
# Each pattern is a tuple of (keyword, weight). Higher weight = stronger signal.
_DEPT_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "hr": [
        ("pto", 2.0), ("paid time off", 2.0), ("vacation", 1.5),
        ("sick leave", 2.0), ("onboarding", 1.5), ("offboarding", 1.5),
        ("handbook", 1.0), ("employee", 1.0), ("benefits", 1.5),
        ("remote work", 1.5), ("work from home", 1.5), ("wfh", 1.5),
        ("parental leave", 2.0), ("maternity", 1.5), ("paternity", 1.5),
        ("performance review", 1.5), ("probation", 1.5), ("termination", 1.0),
        ("hiring", 1.0), ("salary", 1.0), ("compensation", 1.5),
    ],
    "legal": [
        ("nda", 2.0), ("non-disclosure", 2.0), ("contract", 1.5),
        ("vendor", 1.0), ("compliance", 1.0), ("data protection", 2.0),
        ("gdpr", 2.0), ("privacy", 1.5), ("liability", 1.5),
        ("intellectual property", 2.0), ("indemnification", 2.0),
        ("termination clause", 2.0), ("sla", 1.0), ("service level", 1.0),
        ("legal", 1.0), ("regulation", 1.0), ("sox", 1.5),
    ],
    "engineering": [
        ("api", 1.5), ("code review", 2.0), ("pull request", 1.5),
        ("incident", 1.5), ("runbook", 2.0), ("deployment", 1.5),
        ("ci/cd", 2.0), ("git", 1.5), ("testing", 1.0),
        ("architecture", 1.0), ("engineering", 1.0), ("technical", 0.5),
        ("microservice", 1.5), ("endpoint", 1.5), ("versioning", 1.5),
        ("postmortem", 2.0), ("on-call", 1.5), ("pager", 1.5),
    ],
    "finance": [
        ("procurement", 2.0), ("budget", 1.5), ("expense", 1.5),
        ("invoice", 1.5), ("quarterly report", 2.0), ("revenue", 1.5),
        ("cost", 0.5), ("financial", 1.5), ("audit", 1.0),
        ("capex", 2.0), ("opex", 2.0), ("roi", 1.5),
        ("purchase order", 2.0), ("reimbursement", 1.5), ("fiscal", 1.5),
    ],
    "security": [
        ("security policy", 2.0), ("mfa", 2.0), ("multi-factor", 2.0),
        ("password", 1.5), ("access control", 2.0), ("rbac", 2.0),
        ("encryption", 1.5), ("firewall", 1.5), ("vulnerability", 1.5),
        ("phishing", 1.5), ("sso", 1.5), ("single sign-on", 1.5),
        ("security", 1.0), ("breach", 1.5), ("threat", 1.0),
        ("soc", 1.5), ("penetration test", 2.0),
    ],
    "operations": [
        ("business continuity", 2.0), ("bcp", 2.0), ("disaster recovery", 2.0),
        ("change management", 2.0), ("sop", 1.5), ("process", 0.5),
        ("operations", 1.0), ("rto", 1.5), ("rpo", 1.5),
        ("escalation", 1.5), ("stakeholder", 1.0), ("workflow", 1.0),
    ],
}


def detect_department(query: str) -> str | None:
    """Detect the most likely department for a query.

    Returns the department name if confidence is clear (top score is at least
    1.5x the runner-up), otherwise returns None.
    """
    if not query:
        return None

    lower = query.lower()
    scores: dict[str, float] = {}

    for dept, keywords in _DEPT_KEYWORDS.items():
        dept_score = 0.0
        for keyword, weight in keywords:
            if keyword in lower:
                dept_score += weight
        if dept_score > 0:
            scores[dept] = dept_score

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_dept, top_score = ranked[0]

    # Require clear winner: top score >= 1.5 and at least 1.5x runner-up
    if top_score < 1.5:
        return None
    if len(ranked) > 1:
        runner_up_score = ranked[1][1]
        if top_score < runner_up_score * 1.5:
            return None

    return top_dept
