#!/usr/bin/env python3
"""Download and unpack the figure archive, then check it against the manifest.

The archive URL is not baked in: point it at the Zenodo record for the paper
(or any mirror) with --url or the PRESTRESS_DATA_URL environment variable.
See docs/data_release.md.

    python scripts/fetch_results.py --url https://zenodo.org/records/<id>/files/prestress-figures.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("PRESTRESS_DATA_URL", "")


def _progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100 * done / total
    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct:5.1f} %)",
          end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="archive URL (or set PRESTRESS_DATA_URL)")
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--keep-tar", action="store_true")
    args = ap.parse_args()

    if not args.url:
        raise SystemExit(
            "no URL given.  Pass --url, or set PRESTRESS_DATA_URL to the "
            "Zenodo record for the paper (docs/data_release.md).")

    args.dest.mkdir(parents=True, exist_ok=True)
    tar_path = args.dest / "prestress-figures.tar.gz"

    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url) as r, tar_path.open("wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            _progress(done, total)
    print()

    print(f"unpacking into {args.dest}")
    with tarfile.open(tar_path) as tar:
        # filter="data" refuses absolute paths and links pointing outside the
        # destination; the archive is remote, so it is not trusted
        tar.extractall(args.dest, filter="data")

    mf = args.dest / "MANIFEST.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text())
        bad = []
        for name, meta in manifest["files"].items():
            p = args.dest / name
            if not p.is_file():
                bad.append(f"{name}: missing")
                continue
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta["sha256"]:
                bad.append(f"{name}: checksum mismatch")
        if bad:
            raise SystemExit("archive failed verification:\n  "
                             + "\n  ".join(bad))
        print(f"verified {len(manifest['files'])} files against the manifest")

    if not args.keep_tar:
        tar_path.unlink()
    print(f"\nregenerate the figures with:\n"
          f"  make figures DATA={args.dest}")


if __name__ == "__main__":
    main()
