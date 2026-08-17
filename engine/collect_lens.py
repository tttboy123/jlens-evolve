#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = EXPERIMENT_DIR.parent / "jacobian-lens-runtime"
sys.path.insert(0, str(RUNTIME_DIR))

import jlens
from collect import (
    LENS_PATH,
    MODEL_DIR,
    choose_device,
    decode_token,
    top_tokens,
)
from quantitative import compute_concept_metrics, single_token_variants

from lens_features import CONCEPT_GROUPS, build_observation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect score-blind J-lens program signatures."
    )
    parser.add_argument(
        "--trace", type=Path, default=EXPERIMENT_DIR / "runs/main/evolution_trace.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_DIR / "analysis/lens_signatures.jsonl"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum complete observation-prompt length; prompts are never truncated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_trace_edges(path: Path) -> list[dict[str, Any]]:
    by_edge: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not {"parent_id", "child_id", "parent_code", "child_code"} <= row.keys():
            continue
        edge_id = f"{row['parent_id']}->{row['child_id']}"
        if edge_id in by_edge and by_edge[edge_id] != row:
            raise ValueError(f"conflicting duplicate edge: {edge_id}")
        by_edge[edge_id] = row
    return sorted(
        by_edge.values(), key=lambda row: (int(row["iteration"]), row["child_id"])
    )


def program_inventory(edges: list[dict[str, Any]]) -> dict[str, str]:
    programs: dict[str, str] = {}
    for edge in edges:
        for role in ("parent", "child"):
            program_id = str(edge[f"{role}_id"])
            code = str(edge[f"{role}_code"])
            if program_id in programs and programs[program_id] != code:
                raise ValueError(f"conflicting code for program {program_id}")
            programs[program_id] = code
    return programs


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def concept_metrics(
    logits: torch.Tensor,
    *,
    token_groups: dict[str, list[int]],
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    return {
        group: compute_concept_metrics(
            logits,
            concept=group,
            token_variants=token_ids,
            decode=lambda token_id: decode_token(tokenizer, token_id),
        )
        for group, token_ids in token_groups.items()
    }


def collect_program(
    program_id: str,
    code: str,
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    token_groups: dict[str, list[int]],
    max_seq_len: int = 512,
) -> dict[str, Any]:
    prompt = build_observation_prompt(code)
    encoded = model.encode(prompt, max_length=max_seq_len + 1)
    raw_ids = encoded[0].detach().cpu().tolist()
    if len(raw_ids) > max_seq_len:
        raise ValueError(
            f"program {program_id} observation prompt has {len(raw_ids)} tokens; "
            f"limit is {max_seq_len}"
        )
    position = len(raw_ids) - 1
    layers = list(lens.source_layers)
    started = time.perf_counter()
    jlens_logits, model_logits, jlens_ids = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=[position],
        max_seq_len=max_seq_len,
        use_jacobian=True,
    )
    logit_logits, repeated_model_logits, logit_ids = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=[position],
        max_seq_len=max_seq_len,
        use_jacobian=False,
    )
    if not torch.equal(jlens_ids.cpu(), logit_ids.cpu()):
        raise RuntimeError("J-lens and logit-lens tokenization differed")
    torch.testing.assert_close(model_logits, repeated_model_logits, rtol=0, atol=1e-4)

    layer_records = [
        {
            "layer": int(layer),
            "normalized_depth": float(layer / (model.n_layers - 1)),
            "jlens": concept_metrics(
                jlens_logits[layer][0], token_groups=token_groups, tokenizer=tokenizer
            ),
            "logit_lens": concept_metrics(
                logit_logits[layer][0], token_groups=token_groups, tokenizer=tokenizer
            ),
        }
        for layer in layers
    ]
    selected_layers = sorted({layers[0], *layers[::4], layers[-1]})
    layer_top_tokens = {
        str(layer): {
            "jlens": top_tokens(jlens_logits[layer][0], tokenizer, n=5),
            "logit_lens": top_tokens(logit_logits[layer][0], tokenizer, n=5),
        }
        for layer in selected_layers
    }
    return {
        "schema_version": 1,
        "program_id": program_id,
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "token_count": len(raw_ids),
        "observation_max_seq_len": max_seq_len,
        "position": position,
        "position_token": tokenizer.convert_ids_to_tokens(raw_ids)[position],
        "layers": [int(layer) for layer in layers],
        "concept_groups": CONCEPT_GROUPS,
        "layer_records": layer_records,
        "layer_top_tokens": layer_top_tokens,
        "model_output_top_tokens": top_tokens(model_logits[0], tokenizer, n=10),
        "elapsed_seconds": time.perf_counter() - started,
        "collected_at": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    args = parse_args()
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    edges = read_trace_edges(args.trace)
    programs = program_inventory(edges)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed_records = load_records(args.output)
    completed = {str(record["program_id"]) for record in completed_records}
    by_code_hash = {str(record["code_sha256"]): record for record in completed_records}
    pending = [
        (program_id, code)
        for program_id, code in programs.items()
        if program_id not in completed
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pending = pending[: args.limit]
    print(
        f"edges={len(edges)} programs={len(programs)} completed={len(completed)} pending={len(pending)}"
    )
    if not pending:
        return

    device, dtype = choose_device()
    print(f"Loading {MODEL_DIR.name} on {device} with {dtype}", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=dtype, local_files_only=True
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(str(LENS_PATH))

    variants = single_token_variants(
        tokenizer,
        [concept for concepts in CONCEPT_GROUPS.values() for concept in concepts],
    )
    token_groups = {
        group: sorted(
            {token_id for concept in concepts for token_id in variants[concept]}
        )
        for group, concepts in CONCEPT_GROUPS.items()
    }
    unresolved_groups = [
        group for group, token_ids in token_groups.items() if not token_ids
    ]
    if unresolved_groups:
        raise ValueError(
            f"concept groups have no single-token variants: {unresolved_groups}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for index, (program_id, code) in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {program_id}", flush=True)
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            if code_hash in by_code_hash:
                record = {
                    **by_code_hash[code_hash],
                    "program_id": program_id,
                    "reused_from_program_id": by_code_hash[code_hash]["program_id"],
                }
            else:
                record = collect_program(
                    program_id,
                    code,
                    model=model,
                    lens=lens,
                    tokenizer=tokenizer,
                    token_groups=token_groups,
                    max_seq_len=args.max_seq_len,
                )
                by_code_hash[code_hash] = record
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"  tokens={record['token_count']} layers={len(record['layers'])} "
                f"elapsed={record['elapsed_seconds']:.2f}s"
                + (" (cache)" if "reused_from_program_id" in record else ""),
                flush=True,
            )


if __name__ == "__main__":
    main()
