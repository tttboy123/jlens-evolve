"""Generate a patch for a SWE-style task using the local Qwen3.5-4B (MLX).

v2.5 T1 baseline generator: weights frozen, inference only. Produces a unified
diff for one task from its repository checkout + instruction.

Usage:
  python swe_4b_patch.py --checkout <repo-dir> --instruction <text> --out <patch.diff>
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MODEL = "models/Qwen3.5-4B-mlx-4bit"

HUNK_SYSTEM = (
    "You are a patch-generation model. You fix a bug by emitting ONE unified "
    "diff. The tool applies your diff directly; it never reads a full file, so "
    "reproducing a whole file is useless and will be rejected.\n\n"
    "Your reply MUST start with the characters `--- a/`. If you need to think, "
    "think before replying; your visible reply is ONLY the diff.\n\n"
    "Format example (use this SHAPE only; the lines are fictional):\n"
    "--- a/example.py\n+++ b/example.py\n"
    "@@ -1,4 +1,5 @@\n def f(x):\n-    return x\n"
    "+    if x is None:\n+        return 0\n+    return x\n\n"
    "Hard rules:\n"
    "1. Output nothing except the diff. No reasoning, no summary, no markdown "
    "fence, no trailing prose.\n"
    "2. First line must be: --- a/<TARGET_PATH> (the real target path from the "
    "user, NOT example.py).\n"
    "3. Second line must be: +++ b/<TARGET_PATH>\n"
    "4. Then one or more hunks: @@ -A,B +C,D @@ with 3 unchanged context lines "
    "around the -/+ changed lines.\n"
    "5. The -/+ lines MUST be real lines from the target file content below; "
    "never copy the fictional example lines.\n"
    "6. Maximum 3 hunks and 50 changed lines; never exceed 80 total lines.\n"
    "7. Never reproduce the whole file; emit only the hunks that change."
)


def _hunk_user_prompt(instruction: str, targets: list[Path], checkout: Path) -> str:
    """Include target file content (bounded) so the 4B student can make a real
    edit instead of echoing the format example. Large files are excerpted
    around the first symptom-keyword hit; small files are included in full.
    """
    keywords = _clean_keywords(instruction)
    sections: list[str] = []
    for path in targets:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = content.splitlines()
        if len(content) <= 20_000:
            sections.append(f"### {path.relative_to(checkout)}\n\n{content}")
            continue
        low_lines = [line.lower() for line in lines]
        hit = next(
            (i for i, line in enumerate(low_lines) if any(k in line for k in keywords)),
            None,
        )
        if hit is None:
            hit = 0
        lo, hi = max(0, hit - 30), min(len(lines), hit + 30)
        excerpt = "\n".join(
            f"{idx + 1}: {line}" for idx, line in enumerate(lines[lo:hi], start=lo)
        )
        sections.append(
            f"### {path.relative_to(checkout)} (line-numbered excerpt {lo + 1}-{hi}, "
            f"file has {len(lines)} lines)\n\n{excerpt}"
        )
    files_block = "\n\n".join(sections)
    return (
        f"Fix the bug described below.\n\n{instruction.strip()}\n\n"
        f"Target file content (you MUST change lines from this content):\n\n"
        f"{files_block}\n\n"
        "Output ONE unified diff with headers `--- a/<TARGET_PATH>` and "
        "`+++ b/<TARGET_PATH>` (use the exact target path) and hunks "
        "`@@ -A,B +C,D @@` with 3 unchanged context lines around each change. "
        "The removed (-) and added (+) lines MUST come from the real file above; "
        "do not invent placeholder text. Max 3 hunks, 50 changed lines, no full "
        "file, no reasoning."
    )


SYSTEM_EDIT = (
    "You are a senior software engineer. Fix the issue described below in the "
    "given repository. Your output MUST be the COMPLETE corrected content of "
    "the target file(s), byte-for-byte identical to the original except for "
    "the minimal fix. Format: one fenced block per file, first line "
    "```file <relative-path>, last line ```. No explanations, no code "
    "snippets, no other markdown. If a file is long, output every line."
)


SYSTEM = (
    "You are a senior software engineer. Fix the issue described below in the "
    "given repository. Return ONLY the unified diff (git diff format) of your "
    "change inside one fenced code block starting with ```diff. "
    "Do not explain; do not include any other markdown. "
    "Each hunk header (@@ -a,b +c,d @@) must have correct line counts."
)

MAX_CONTENT_CHARS = 120_000
MAX_SINGLE_FILE = 100_000
MAX_FILES = 3

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "has",
        "had",
        "have",
        "but",
        "not",
        "you",
        "its",
        "all",
        "can",
        "will",
        "into",
        "when",
        "your",
        "been",
        "than",
        "then",
        "there",
        "they",
        "their",
        "what",
        "which",
        "would",
        "could",
        "should",
        "about",
        "after",
        "before",
        "because",
        "only",
        "also",
        "more",
        "most",
        "some",
        "such",
        "may",
        "might",
        "does",
        "did",
        "while",
        "why",
        "how",
        "where",
        "who",
        "whom",
        "sphinx",
        "issue",
        "fix",
        "bug",
        "fixes",
        "close",
        "closes",
        "added",
        "add",
        "support",
        "show",
        "shows",
        "new",
        "using",
        "use",
        "used",
        "via",
        "etc",
        "please",
        "make",
        "makes",
        "made",
        "now",
        "even",
        "well",
        "without",
        "within",
        "across",
        "each",
        "other",
        "both",
        "between",
        "under",
        "over",
        "any",
        "every",
        "few",
        "many",
        "much",
        "still",
        "yet",
        "output",
        "result",
        "describe",
        "reproduce",
        "build",
        "expected",
        "actual",
        "behavior",
        "additional",
        "steps",
        "screenshot",
        "attached",
        "report",
        "to",
        "reproducible",
        "current",
        "observed",
    }
)

_SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".rb",
    ".php",
    ".cs",
    ".kt",
    ".swift",
    ".sh",
}

_EXCLUDE_PARTS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    "minified",
    "locale",
    ".github",
    "roots",
}


def _clean_keywords(instruction: str, limit: int = 8) -> list[str]:
    """Extract real symptom keywords: strip markdown/code/urls and stopwords."""
    text = re.sub(r"`[^`]*`", " ", instruction or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9_]", " ", text)
    seen: list[str] = []
    for word in text.split():
        low = word.lower()
        if len(low) <= 2 or low in _STOPWORDS or low.isdigit():
            continue
        if low not in seen:
            seen.append(low)
        if len(seen) >= limit:
            break
    return seen


def _stem(keyword: str) -> str:
    """Light suffix-stripping so issue prose matches code identifiers
    (e.g. ``mocked`` -> ``mock``, ``inherited`` -> ``inherit``)."""
    if len(keyword) > 4 and keyword.endswith("ed"):
        return keyword[:-2]
    if len(keyword) > 4 and keyword.endswith("ing"):
        return keyword[:-3]
    if len(keyword) > 4 and keyword.endswith("es"):
        return keyword[:-2]
    if len(keyword) > 4 and keyword.endswith("s"):
        return keyword[:-1]
    return keyword


def _relevant_files(checkout: Path, instruction: str) -> list[Path]:
    """Pick files most relevant to the issue.

    Scoring is content-first (keyword appears in file body) with a bounded
    scan, then repo-relative path matches as tiebreak. Fixes two pipeline
    bugs: Python sources were excluded (only .js/.ts/.json/.md), and keywords
    were naive tokens from raw Markdown matched against the absolute path.
    """
    keywords = _clean_keywords(instruction)
    root = checkout.resolve()
    content_hits: dict[Path, set[str]] = {}
    path_hits: dict[Path, set[str]] = {}
    scanned = 0
    scanned_files = 0
    scan_cap = 8_000_000
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        if {part.lower() for part in path.parts} & _EXCLUDE_PARTS:
            continue
        rel = str(path.relative_to(root)).lower()
        ph = {
            k for k in keywords if k in rel or (len(_stem(k)) >= 3 and _stem(k) in rel)
        }
        if ph:
            path_hits[path] = ph
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 250_000 or scanned + size > scan_cap:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        scanned += size
        scanned_files += 1
        ch = {k for k in keywords if k in content}
        if ch:
            content_hits[path] = ch
    # Keep only discriminative keywords: present in few scanned files.
    # Generic terms ("code", "start", "end", "default", "value") drown the
    # signal; symptoms like "positional"/"whitespace"/"highlighting" survive.
    df: dict[str, int] = {}
    for hits in content_hits.values():
        for k in hits:
            df[k] = df.get(k, 0) + 1
    for hits in path_hits.values():
        for k in hits:
            df[k] = df.get(k, 0) + 1
    rare_limit = max(3, int(0.12 * max(1, scanned_files)))
    rare = {k for k in keywords if 1 <= df.get(k, 0) <= rare_limit}
    content_hits = {
        path: hits & rare for path, hits in content_hits.items() if hits & rare
    }
    path_hits = {path: hits & rare for path, hits in path_hits.items() if hits & rare}

    def weight(keyword: str) -> float:
        return 1.0 / (1.0 + df.get(keyword, 0))

    def suffix_weight(path: Path) -> float:
        # source code dominates docs/templates; .md/.json are weak signals
        return 0.15 if path.suffix in {".md", ".json"} else 1.0

    merged: dict[Path, float] = {}
    for path, hits in content_hits.items():
        merged[path] = merged.get(path, 0.0) + 3.0 * suffix_weight(path) * sum(
            weight(k) for k in hits
        )
    for path, hits in path_hits.items():
        merged[path] = merged.get(path, 0.0) + 5.0 * suffix_weight(path) * sum(
            weight(k) for k in hits
        )
    if not merged:
        return []
    ordered = sorted(merged, key=lambda path: (-merged[path], str(path)))
    files: list[Path] = []
    total = 0
    for path in ordered:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SINGLE_FILE or total + size > MAX_CONTENT_CHARS:
            continue
        files.append(path)
        total += size
        if len(files) >= MAX_FILES:
            break
    return files


def generate_patch(checkout: Path, instruction: str) -> str:
    from mlx_lm import generate, load

    files = _relevant_files(checkout, instruction)
    if not files:
        files = [path for path in sorted(checkout.rglob("*")) if path.is_file()][:3]
    sections = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sections.append(f"### {path.relative_to(checkout)}\n\n{content}")
    repo_context = "\n\n".join(sections)
    model, tokenizer = load(MODEL)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Repository: {checkout.name}\nIssue: {instruction}\n\n"
                f"Relevant source files:\n\n{repo_context}\n\n"
                "Return ONLY the unified diff (git diff format) that fixes the issue."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=4096)
    return text


def apply_and_diff(checkout: Path, patch_text: str) -> str:
    """Extract and validate the unified diff from raw model output.

    Model output often contains reasoning prose and markdown fences. We extract
    the fenced ```diff block when present, otherwise the text starting at the
    first ``diff --git`` line, then require the result to apply cleanly against
    the checkout (``git apply --check``). Raw output is preserved by the caller
    as negative evidence when extraction/validation fails.
    """
    candidate: str | None = None
    fenced = re.search(r"```diff\s*\n(.*?)```", patch_text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip() + "\n"
    else:
        start = patch_text.find("diff --git")
        if start >= 0:
            candidate = patch_text[start:].rstrip() + "\n"
    if candidate is None:
        raise ValueError("model output contains no diff block")

    # Progressive trailing-trim fallback: drop trailing prose lines that follow
    # the diff until `git apply --check` accepts the remainder.
    lines = candidate.splitlines()
    while len(lines) >= 10:
        probe = "\n".join(lines) + "\n"
        if _applies_cleanly(checkout, probe):
            return probe
        lines = lines[:-1]
    raise ValueError("extracted diff does not apply cleanly to checkout")


def _extract_hunk(text: str) -> tuple[str | None, str]:
    """Extract a small unified diff from model output.

    Returns (diff, rejection_reason): reason is None when a diff block was
    found and validated; otherwise one of no-diff / bad-header /
    malformed-hunk / apply-fail. Leading reasoning is tolerated (FM1 fix):
    we start at the first line beginning with ``--- a/``.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("--- a/")), None)
    if start is None:
        return None, "no-diff"
    diff_lines = lines[start:]
    # drop trailing prose after the last hunk marker
    end = max(
        (i for i, line in enumerate(diff_lines) if line.startswith("@@")),
        default=None,
    )
    if end is None:
        return None, "malformed-hunk"
    # keep until the end of the final hunk: after @@ line + body; find next
    # non-hunk prose by requiring a plausible hunk body (context/-/+ lines).
    body_end = len(diff_lines)
    for i in range(end, len(diff_lines)):
        line = diff_lines[i]
        if i > end and not (
            line.startswith((" ", "+", "-", "@@", "---", "+++"))
            or line == "\\ No newline at end of file"
        ):
            body_end = i
            break
    candidate = "\n".join(diff_lines[:body_end]).rstrip() + "\n"
    if "+++ b/" not in candidate:
        return None, "bad-header"
    return candidate, None


def generate_hunk(
    checkout: Path, instruction: str, skill: str | None = None, max_tokens: int = 300
) -> str:
    """Diff-snippet protocol: student outputs a small unified diff."""
    from mlx_lm import generate, load

    targets, context_files = _pick_target_files(checkout, instruction)
    if not targets:
        raise ValueError("no editable target file found")
    model, tokenizer = load(MODEL)
    skill_block = (
        "\n\n### Teaching skill (follow it)\n" + skill.strip()
        if skill and skill.strip()
        else ""
    )
    system = HUNK_SYSTEM + skill_block
    user = _hunk_user_prompt(instruction, targets, checkout)
    if context_files:
        user += "\n\nRead-only test hint (do NOT edit tests): " + ", ".join(
            str(f.relative_to(checkout)) for f in context_files
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def apply_and_diff_hunk(checkout: Path, raw_text: str) -> tuple[str, str | None]:
    """Validate hunk-mode output; returns (diff, rejection_reason)."""
    diff, reason = _extract_hunk(raw_text)
    if diff is None:
        return "", reason
    if not _applies_cleanly(checkout, diff):
        # tolerant fallback: try patch -p1 --fuzz=2 on a temp worktree? keep
        # simple: report apply-fail unless git apply already validated.
        return diff, "apply-fail"
    return diff, None


def _applies_cleanly(checkout: Path, diff: str) -> bool:
    proc = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=checkout,
        input=diff,
        capture_output=True,
        check=False,
        text=True,
    )
    return proc.returncode == 0


def _pick_target_files(
    checkout: Path, instruction: str
) -> tuple[list[Path], list[Path]]:
    """Return (edit_targets, context_files) for edit mode.

    edit_targets: SOURCE files the model may modify (tests excluded so the eval
    harness stays authoritative). context_files: best-matching TEST files shown
    read-only, so the model can locate expected behavior without gaming tests.
    """
    files = _relevant_files(checkout, instruction)
    if not files:
        files = [path for path in sorted(checkout.rglob("*")) if path.is_file()][:3]
    edit_targets: list[Path] = []
    context_files: list[Path] = []
    for f in files:
        parts = f.parts
        is_test = any(part in {"test", "tests", "__tests__"} for part in parts) or (
            ".test." in f.name.lower() or ".spec." in f.name.lower()
        )
        if is_test:
            if len(context_files) < 1 and f.stat().st_size <= 150_000:
                context_files.append(f)
            continue
        if len(edit_targets) < 2 and f.stat().st_size <= 150_000:
            edit_targets.append(f)
    if not edit_targets:
        raise ValueError("no non-test target file found for instruction")
    return edit_targets, context_files


def generate_edit(checkout: Path, instruction: str, skill: str | None = None) -> str:
    """Edit-then-diff variant: model outputs the corrected full file; tool diffs."""
    from mlx_lm import generate, load

    targets, context_files = _pick_target_files(checkout, instruction)
    if not targets:
        raise ValueError("no editable target file found")
    model, tokenizer = load(MODEL)
    sections = []
    for path in context_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sections.append(
            f"### Test file (READ-ONLY context, do NOT edit): {path.relative_to(checkout)}\n\n{content}"
        )
    for path in targets:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sections.append(f"### {path.relative_to(checkout)}\n\n{content}")
    repo_context = "\n\n".join(sections)
    skill_block = (
        "\n\n### Teaching skill (follow it when it applies)\n" + skill.strip()
        if skill and skill.strip()
        else ""
    )
    messages = [
        {"role": "system", "content": SYSTEM_EDIT},
        {
            "role": "user",
            "content": (
                f"Repository: {checkout.name}\nIssue: {instruction}\n\n"
                f"Target file(s):\n\n{repo_context}\n" + skill_block + "\n\n"
                "Output the COMPLETE corrected content of each target file, "
                "each inside a fenced block starting with ```file <relative-path>. "
                "Preserve all unchanged lines exactly, every line. Do not explain."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=8192)
    return text


def edits_to_diff(checkout: Path, raw_text: str) -> str:
    """Extract fenced file blocks, write edited files, build a unified diff."""
    import tempfile

    blocks = re.findall(r"```file\s+([^\n\s]+)\s*\n(.*?)```", raw_text, re.DOTALL)
    if not blocks:
        raise ValueError("edit output contains no fenced file blocks")
    root = checkout.resolve()
    diffs: list[str] = []
    with tempfile.TemporaryDirectory(dir="/private/tmp/swe-4b") as tmp:
        for rel, content in blocks:
            src = (root / rel).resolve()
            if not str(src).startswith(str(root)):
                continue
            if not src.is_file():
                raise ValueError(f"target not in checkout: {rel}")
            edited = Path(tmp) / rel
            edited.parent.mkdir(parents=True, exist_ok=True)
            edited.write_text(content.strip() + "\n", encoding="utf-8")
        for rel, _content in blocks:
            src = (root / rel).resolve()
            if not str(src).startswith(str(root)) or not src.is_file():
                continue
            edited = Path(tmp) / rel
            proc = subprocess.run(
                ["git", "diff", "--no-index", "--", str(src), str(edited)],
                capture_output=True,
                check=False,
                text=True,
            )
            diff = proc.stdout
            if not diff.strip():
                continue
            rel_posix = src.relative_to(root).as_posix()
            diff = diff.replace(f"a{src}", f"a/{rel_posix}").replace(
                f"b{edited}", f"b/{rel_posix}"
            )
            diffs.append(diff)
    cleaned = "\n".join(diffs).strip() + "\n"
    if not cleaned.strip():
        raise ValueError("edit output produced no diff (file unchanged)")
    if not _applies_cleanly(checkout, cleaned):
        raise ValueError("edit-then-diff does not apply cleanly to checkout")
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["diff", "edit", "hunk"],
        default="diff",
        help="diff: raw unified diff. edit: corrected full file, tool diffs. hunk: small diff-snippet protocol (DeepSeek v2 teaching plan).",
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=None,
        help="path to a teaching SKILL.md to inject into the edit prompt",
    )
    args = parser.parse_args()
    if not args.checkout.is_dir():
        raise SystemExit(f"checkout not found: {args.checkout}")
    skill_text = args.skill.read_text(encoding="utf-8") if args.skill else None
    if args.mode == "hunk":
        raw = generate_hunk(args.checkout, args.instruction, skill=skill_text)
        diff, reason = apply_and_diff_hunk(args.checkout, raw)
        if reason is not None:
            raw_path = args.out.with_suffix(args.out.suffix + ".raw.txt")
            raw_path.write_text(raw, encoding="utf-8")
            print(f"HUNK REJECTED ({reason}); raw kept at {raw_path}")
            return 2
        args.out.write_text(diff, encoding="utf-8")
        print(f"hunk patch bytes: {len(diff)} -> {args.out}")
        return 0
    if args.mode == "edit":
        raw = generate_edit(args.checkout, args.instruction, skill=skill_text)
        try:
            patch = edits_to_diff(args.checkout, raw)
        except ValueError as exc:
            raw_path = args.out.with_suffix(args.out.suffix + ".raw.txt")
            raw_path.write_text(raw, encoding="utf-8")
            print(f"INVALID EDIT: {exc}; raw output kept at {raw_path}")
            return 2
        args.out.write_text(patch, encoding="utf-8")
        print(f"edit patch bytes: {len(patch)} -> {args.out}")
        return 0
    text = generate_patch(args.checkout, args.instruction)
    try:
        patch = apply_and_diff(args.checkout, text)
    except ValueError as exc:
        raw_path = args.out.with_suffix(args.out.suffix + ".raw.txt")
        raw_path.write_text(text, encoding="utf-8")
        print(f"INVALID PATCH: {exc}; raw output kept at {raw_path}")
        return 2
    args.out.write_text(patch, encoding="utf-8")
    print(f"patch bytes: {len(patch)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
