import hashlib
import json
from pathlib import Path

from artifact_verifier import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v1.0.0/MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_verifier_is_read_only_and_checks_every_listed_file():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed_before = {
        str(MANIFEST): _sha256(MANIFEST),
        str(ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/DECISION.json"): _sha256(
            ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/DECISION.json"
        ),
    }

    report = verify_manifest(MANIFEST)

    assert report["valid"]
    assert report["files_verified"] == sum(
        len(stage["files"]) for stage in data["stages"]
    )
    assert report["stages_verified"] == len(data["stages"])
    assert report["failures"] == []
    assert observed_before == {path: _sha256(Path(path)) for path in observed_before}


def test_manifest_verifier_rejects_wrong_hash_without_touching_target(tmp_path: Path):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    target = ROOT / "artifacts/v1.0.0" / data["stages"][0]["files"][0]["path"]
    before = _sha256(target)
    data["stages"][0]["files"][0]["sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text(json.dumps(data), encoding="utf-8")

    report = verify_manifest(bad_manifest)

    assert not report["valid"]
    assert report["failures"][0]["reason"] == "sha256_mismatch"
    assert _sha256(target) == before


def test_manifest_verifier_rejects_path_traversal(tmp_path: Path):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["stages"][0]["files"][0]["path"] = "../outside.txt"
    bad_manifest = tmp_path / "traversal.json"
    bad_manifest.write_text(json.dumps(data), encoding="utf-8")

    report = verify_manifest(bad_manifest)

    assert not report["valid"]
    assert report["failures"][0]["reason"] == "unsafe_path"
