#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".txt",
    ".log",
    ".md",
}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt"}


def rel_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Pack toy experiment results for inspection")
    p.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories")
    p.add_argument("--out", default=None, help="Output zip path; default uses timestamp")
    p.add_argument("--include-models", action="store_true", help="Include model_final.pt/checkpoints; may make zip larger")
    p.add_argument("--extra", nargs="*", default=[], help="Extra files or directories to include")
    args = p.parse_args()

    root = Path.cwd()
    runs_dir = Path(args.runs_dir)
    if args.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path(f"cp_toy_results_{stamp}.zip")
    else:
        out = Path(args.out)

    include_suffixes = set(DEFAULT_SUFFIXES)
    if args.include_models:
        include_suffixes |= MODEL_SUFFIXES

    candidates: list[Path] = []
    if runs_dir.exists():
        candidates.extend([p for p in runs_dir.rglob("*") if p.is_file()])
    else:
        print(f"warning: runs dir {runs_dir} does not exist", file=sys.stderr)

    for extra in args.extra:
        ep = Path(extra)
        if ep.is_dir():
            candidates.extend([p for p in ep.rglob("*") if p.is_file()])
        elif ep.is_file():
            candidates.append(ep)
        else:
            print(f"warning: extra path {ep} does not exist", file=sys.stderr)

    selected: list[Path] = []
    for f in candidates:
        if f.suffix in include_suffixes:
            selected.append(f)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(root),
        "runs_dir": str(runs_dir),
        "include_models": args.include_models,
        "git_status": command_output(["git", "status", "--short"]),
        "git_rev": command_output(["git", "rev-parse", "HEAD"]),
        "files": [],
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(set(selected)):
            try:
                arc = rel_to(f, root)
            except ValueError:
                arc = f.name
            zf.write(f, arc)
            manifest["files"].append({"path": arc, "bytes": f.stat().st_size})
        zf.writestr("RESULTS_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))

    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"included {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
