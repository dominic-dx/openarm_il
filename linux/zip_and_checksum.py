#!/usr/bin/env python3
"""
Run on the LAPTOP after package_dataset.py finishes.
Zips the packaged dataset and writes a DONE marker + checksum so you
can verify the transfer over SSH before/after copying.

Usage:
    python3 zip_and_checksum.py --input ./packaged_dataset --output ./packaged_dataset.zip
"""

import argparse
import hashlib
import shutil
import time
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_and_mark(input_dir: Path, output_zip: Path):
    status_file = input_dir.parent / "status.log"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(status_file, "a") as f:
            f.write(line + "\n")

    log("ZIPPING started")
    base_name = str(output_zip.with_suffix(""))
    shutil.make_archive(base_name, "zip", root_dir=input_dir)
    log(f"ZIPPING complete: {output_zip}")

    checksum = sha256_of_file(output_zip)
    checksum_path = output_zip.with_suffix(".sha256")
    checksum_path.write_text(checksum + "\n")

    done_marker = output_zip.parent / "TRANSFER_READY"
    done_marker.write_text(f"zip={output_zip.name}\nsha256={checksum}\ntimestamp={time.time()}\n")

    log(f"DONE. sha256={checksum}")
    log(f"Marker written: {done_marker}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    zip_and_mark(args.input, args.output)