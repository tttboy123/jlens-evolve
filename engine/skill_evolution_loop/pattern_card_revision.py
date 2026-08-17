"""Bounded feedback-only refinement of inactive PatternCard trigger prose."""

from __future__ import annotations

import re

from .contracts import ContractError, LoopRevision, sha256_json
from .mlx_student import _pattern_cards

_PROHIBITED = re.compile(
    r"(?:p1-|round1-|[/\\]|\.(?:py|js|rs|java|php)\b)", re.IGNORECASE
)


def refine_pattern_card_symptoms(
    *,
    parent: LoopRevision,
    replacements: dict[int, str],
    source_evidence_sha256: str,
    revision_id: str,
) -> dict[str, object]:
    """Create an inactive child changing only complete PatternCard symptoms."""

    parent.validate()
    cards = _pattern_cards(parent.skill_text)
    if not replacements:
        raise ContractError("PatternCard symptom replacements must be non-empty")
    available = set(range(1, len(cards) + 1))
    if set(replacements) - available:
        raise ContractError("replacement is outside the available PatternCards")
    if not re.fullmatch(r"[0-9a-f]{64}", source_evidence_sha256):
        raise ContractError("invalid source evidence sha256")

    revised = parent.skill_text
    normalized: dict[int, str] = {}
    for card_number, symptom in sorted(replacements.items()):
        value = " ".join(symptom.strip().split())
        if not value:
            raise ContractError("PatternCard symptom replacement must be non-empty")
        if len(value) > 300:
            raise ContractError("PatternCard symptom replacement is too long")
        if _PROHIBITED.search(value):
            raise ContractError(
                "PatternCard symptoms cannot encode task IDs or file paths"
            )
        card = cards[card_number - 1]
        replacement = re.sub(
            r"(?s)^(\d+\.\s+Symptom:\s*).*?(\s+Transformation:\s*)",
            lambda match: f"{match.group(1)}{value}.{match.group(2)}",
            card,
            count=1,
        )
        if replacement == card:
            raise ContractError(
                "PatternCard symptom replacement did not change the card"
            )
        if card.count("Transformation:") != 1 or card.count("Validation:") != 1:
            raise ContractError("PatternCard structure is ambiguous")
        revised = revised.replace(card, replacement, 1)
        normalized[card_number] = value

    if "active: false" not in revised or "auto_install: false" not in revised:
        raise ContractError("PatternCard parent is not explicitly inactive")
    child = LoopRevision.create(
        skill_id=parent.skill_id,
        revision_id=revision_id,
        parent_revision_id=parent.revision_id,
        source_round=parent.source_round + 1,
        protocol=parent.protocol,
        skill_text=revised,
        prompt_template=parent.prompt_template,
        eval_note=(
            "Feedback-only PatternCard symptom-anchor refinement; Transformation and "
            "Validation clauses preserved; inactive and unpromoted."
        ),
    )
    content: dict[str, object] = {
        "schema_version": 1,
        "candidate_source": "feedback-pattern-card-trigger-refinement-v1",
        "source_evidence_sha256": source_evidence_sha256,
        "replaced_card_numbers": sorted(normalized),
        "next_revision": child.to_dict(),
        "candidate_status": "inactive",
        "auto_activate": False,
        "holdout_task_ids_included": False,
        "network_calls_performed": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}
