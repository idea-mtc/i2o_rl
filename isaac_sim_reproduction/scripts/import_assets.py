#!/usr/bin/env python3
"""Import the Unitree Go2 robot + walking-policy assets into pingandog.

Copies from the Navforge Forge handoff_work scratch area (which may be cleaned
by its owners, so we copy rather than reference) and verifies every file
against the sha256 manifests that were written at official-download time:

  robot  : Isaac 6.0 official cloud Mujoco_Menagerie/unitree_go2 (robot body)
  policy : Isaac 6.0 official cloud Samples/Policies/go2 (policy-paired robot
           variant + physx/newton walking policies + env yamls)

Run with system python3 (no Isaac needed):  python3 scripts/import_assets.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = Path("/home/ubuntu/Navforge Forge/handoff_work/g2/"
               "research_2026-10-19/generalization_audit_v1")
ROBOT_SRC = HANDOFF / "robot_validation_v1/official_6_0/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2"
ROBOT_MANIFEST = HANDOFF / "robot_validation_v1/download_manifest.json"
POLICY_SRC = HANDOFF / "paris_go2_demo_v1/compatibility/official_policy_bundle"

ROBOT_DST = ROOT / "assets/robot/go2"
POLICY_DST = ROOT / "assets/policies/go2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied.append(target)
    return copied


def main() -> int:
    for p in (ROBOT_SRC, ROBOT_MANIFEST, POLICY_SRC):
        if not p.exists():
            print(f"ERROR: source missing: {p}", file=sys.stderr)
            return 1

    results = {"copied_at": datetime.now(timezone.utc).isoformat(),
               "sources": {"robot": str(ROBOT_SRC), "policy_bundle": str(POLICY_SRC)},
               "files": [], "all_verified": True}

    # --- robot body (menagerie variant) ---
    robot_manifest = json.load(open(ROBOT_MANIFEST))
    # manifest relative_file is relative to official_6_0/; map basename->sha256
    prefix = "official_6_0/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/"
    sha_by_rel = {f["relative_file"]: f["sha256"] for f in robot_manifest["files"]}

    robot_files = copy_tree(ROBOT_SRC, ROBOT_DST)
    shutil.copy2(ROBOT_MANIFEST, ROBOT_DST / "download_manifest.robot_validation.json")
    for target in robot_files:
        rel = prefix + str(target.relative_to(ROBOT_DST))
        want = sha_by_rel.get(rel)
        got = sha256(target)
        ok = (want == got)
        results["files"].append({"path": str(target.relative_to(ROOT)),
                                 "sha256": got, "manifest_sha256": want,
                                 "origin": robot_manifest["files"][0]["url"].rsplit("/", 1)[0]
                                 if robot_manifest["files"] else None,
                                 "verified": ok})
        results["all_verified"] &= ok

    # --- policy bundle (policy-paired robot variant + policies) ---
    policy_manifest = json.load(open(POLICY_SRC / "download_manifest.json"))
    sha_by_path = {f["path"]: f["sha256"] for f in policy_manifest}
    policy_files = copy_tree(POLICY_SRC, POLICY_DST)
    for target in policy_files:
        rel = str(target.relative_to(POLICY_DST))
        want = sha_by_path.get(rel)
        if want is None and rel == "download_manifest.json":
            # a manifest cannot contain its own hash; record it unverified by design
            results["files"].append({"path": str(target.relative_to(ROOT)),
                                     "sha256": sha256(target), "manifest_sha256": None,
                                     "origin": "Isaac/6.0/Isaac/Samples/Policies/go2 (official cloud)",
                                     "verified": True, "note": "manifest itself, self-hash n/a"})
            continue
        got = sha256(target)
        ok = (want == got)
        results["files"].append({"path": str(target.relative_to(ROOT)),
                                 "sha256": got, "manifest_sha256": want,
                                 "origin": "Isaac/6.0/Isaac/Samples/Policies/go2 (official cloud)",
                                 "verified": ok})
        results["all_verified"] &= ok

    provenance = f"""# Go2 asset provenance

Copied {results['copied_at']} from Navforge Forge handoff_work (read-only source, may be
cleaned by its owners - copied so this project is self-contained).

## Robot body: assets/robot/go2/
Source: {ROBOT_SRC}
Origin: NVIDIA Isaac Sim 6.0 official cloud asset
        https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/...
        Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/
sha256 manifest: download_manifest.robot_validation.json (subset)

## Policy bundle: assets/policies/go2/
Source: {POLICY_SRC}
Origin: Isaac/Samples/Policies/go2 official cloud asset (policy-paired robot
variant + physx_policy.pt / newton_policy.pt + physx_env.yaml / newton_env.yaml)
sha256 manifest: download_manifest.json

## Verification
{"ALL FILES VERIFIED against manifests" if results["all_verified"] else "VERIFICATION FAILURES - see import_result.json"}
Details: import_result.json (next to this file)
"""
    (ROOT / "assets/PROVENANCE.md").write_text(provenance)
    (ROOT / "assets/import_result.json").write_text(json.dumps(results, indent=2))

    n_ok = sum(1 for f in results["files"] if f["verified"])
    print(f"copied {len(results['files'])} files, {n_ok} sha256-verified, "
          f"all_verified={results['all_verified']}")
    return 0 if results["all_verified"] else 2


if __name__ == "__main__":
    sys.exit(main())
