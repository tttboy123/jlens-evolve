"""Zero-network fallback that compiles frozen feedback into an inactive Skill."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .p1_parent import load_frozen_p1_parent_revision
from .registry import LoopRevisionRegistry

_PROHIBITED = re.compile(r"(?:p1-|[/\\]|\.(?:py|js|rs)\b)", re.IGNORECASE)


def freeze_p1_local_feedback_revision(
    *,
    current_revision_path: Path,
    semantic_review_path: Path,
    pattern_cards: list[str],
    output_path: Path,
    registry_root: Path,
) -> dict[str, object]:
    """Compile anonymous feedback patterns without another network call."""
    current = load_frozen_p1_parent_revision(current_revision_path)
    if current.source_round < 1:
        raise ContractError("local feedback compiler requires a parent revision")
    try:
        semantic_raw = semantic_review_path.read_bytes()
        semantic = json.loads(semantic_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("P1 semantic review is unreadable") from exc
    if (
        not isinstance(semantic, dict)
        or semantic.get("network_calls_performed") is not False
    ):
        raise ContractError("P1 semantic review must be a frozen offline object")
    if semantic.get("skill_revision_fingerprint") != current.fingerprint:
        raise ContractError("P1 semantic review revision mismatch")
    if not 1 <= len(pattern_cards) <= 3:
        raise ContractError("local feedback compiler requires one to three cards")
    normalized = [card.strip() for card in pattern_cards]
    if any(not card or len(card) > 600 for card in normalized):
        raise ContractError("local feedback pattern card is empty or too long")
    if any(_PROHIBITED.search(card) for card in normalized):
        raise ContractError("local feedback cards cannot encode task IDs or file paths")
    cards = "\n".join(f"{index}. {card}" for index, card in enumerate(normalized, 1))
    skill_text = (
        "---\n"
        "name: local-feedback-pattern-cards\n"
        "project_local_only: true\n"
        "auto_install: false\n"
        "active: false\n"
        "---\n\n"
        "Return only {file, search, replace, diagnostic} JSON. Select the one "
        "pattern whose symptom and data flow match the issue; do not combine cards.\n\n"
        "## Pattern cards\n"
        f"{cards}\n\n"
        "## Commit gate\n"
        "Copy an exact unique search span. Make one minimal replacement. Reject "
        "no-op, duplicate, unreachable, invented-API, and opposite-direction edits. "
        "Simulate the reported edge case and one ordinary path before returning JSON."
    )
    if len(skill_text) > 2500:
        raise ContractError("compiled local feedback Skill exceeds 2500 characters")
    source_round = current.source_round + 1
    revision = LoopRevision.create(
        skill_id=current.skill_id,
        revision_id=f"{current.skill_id}-r{source_round:03d}-local-patterns",
        parent_revision_id=current.revision_id,
        source_round=source_round,
        protocol=current.protocol,
        skill_text=skill_text,
        prompt_template=current.prompt_template,
        eval_note=(
            "Zero-network fallback compiled from frozen feedback after the 5/5 "
            "parent-call boundary; observational, inactive, and unpromoted."
        ),
    )
    content = {
        "schema_version": 1,
        "candidate_source": "local-feedback-compiler-v1",
        "source_semantic_review_sha256": hashlib.sha256(semantic_raw).hexdigest(),
        "pattern_card_count": len(normalized),
        "holdout_task_ids_included": False,
        "next_revision": revision.to_dict(),
        "candidate_status": "inactive",
        "auto_activate": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                "local feedback revision evidence is unreadable"
            ) from exc
        if existing != report:
            raise ContractError("frozen local feedback revision does not match")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    registry = LoopRevisionRegistry(registry_root)
    latest = registry.latest(current.skill_id)
    if latest is None or latest.fingerprint != current.fingerprint:
        if latest is None or latest.fingerprint != revision.fingerprint:
            raise ContractError("P1 registry head does not match local compiler input")
    registry.append(revision)
    return report


def load_frozen_p1_local_feedback_revision(path: Path) -> LoopRevision:
    """Load a local fallback only after its inactive/no-network evidence validates."""
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("local feedback revision evidence is unreadable") from exc
    if not isinstance(wrapper, dict):
        raise ContractError("local feedback revision evidence must be an object")
    evidence_sha = wrapper.get("evidence_sha256")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError("local feedback revision evidence sha256 mismatch")
    if (
        wrapper.get("candidate_source") != "local-feedback-compiler-v1"
        or wrapper.get("holdout_task_ids_included") is not False
        or wrapper.get("candidate_status") != "inactive"
        or wrapper.get("auto_activate") is not False
        or wrapper.get("network_calls_performed") is not False
    ):
        raise ContractError("local feedback revision boundary is invalid")
    revision = LoopRevision.from_dict(wrapper.get("next_revision"))
    if (
        "active: false" not in revision.skill_text
        or "auto_install: false" not in revision.skill_text
    ):
        raise ContractError("local feedback revision is not explicitly inactive")
    return revision
