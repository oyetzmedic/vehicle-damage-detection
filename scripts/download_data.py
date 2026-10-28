from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from roboflow import Roboflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY in your environment before downloading the dataset."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        if any(args.output.iterdir()):
            if not args.overwrite:
                raise SystemExit(
                    f"{args.output} is not empty. Re-run with --overwrite to replace it."
                )
            shutil.rmtree(args.output)
        else:
            # Roboflow's SDK treats any existing location as already downloaded
            # unless overwrite=True, so remove an empty directory first.
            args.output.rmdir()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("brigada4").project("damage-on-cars")
    project.version(args.version).download(
        "coco",
        location=str(args.output),
        overwrite=args.overwrite,
    )

    annotation_files = list(args.output.rglob("_annotations.coco.json"))
    if not annotation_files:
        raise RuntimeError(
            f"Roboflow returned without a usable COCO export in {args.output}."
        )

    print(f"Downloaded Roboflow COCO export to {args.output}")
    print(f"Found {len(annotation_files)} COCO annotation file(s).")
    print("Run scripts/prepare_data.py next to filter classes and cap the dataset.")


if __name__ == "__main__":
    main()
