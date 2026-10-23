from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboflow import Roboflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--version", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY in your environment before downloading the dataset."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("brigada4").project("damage-on-cars")
    project.version(args.version).download("coco", location=str(args.output))

    print(f"Downloaded Roboflow COCO export to {args.output}")
    print("Run scripts/prepare_data.py next to filter classes and cap the dataset.")


if __name__ == "__main__":
    main()
