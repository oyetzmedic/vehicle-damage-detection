from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

TARGET_CLASSES = ("Crack", "Scratch", "Dent", "Broken Glass")


def load_split(split_dir: Path):
    annotation_path = split_dir / "_annotations.coco.json"
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-images", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_names = ("train", "valid", "test")
    split_data = {}
    eligible = []

    for split in split_names:
        split_dir = args.raw / split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue

        data = load_split(split_dir)
        categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        target_category_ids = {
            category_id
            for category_id, name in categories.items()
            if name in TARGET_CLASSES
        }
        target_image_ids = {
            ann["image_id"]
            for ann in data["annotations"]
            if ann["category_id"] in target_category_ids
        }
        split_data[split] = (data, categories, target_category_ids)
        eligible.extend((split, image["id"]) for image in data["images"] if image["id"] in target_image_ids)

    if not eligible:
        raise SystemExit(f"No eligible images found under {args.raw}")

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = set(eligible[: min(args.max_images, len(eligible))])

    if args.output.exists():
        shutil.rmtree(args.output)

    totals = {}
    for split, (data, categories, target_category_ids) in split_data.items():
        output_split = args.output / split
        output_split.mkdir(parents=True, exist_ok=True)

        kept_images = [
            image for image in data["images"] if (split, image["id"]) in selected
        ]
        kept_ids = {image["id"] for image in kept_images}

        source_name_to_id = {
            name: category_id
            for category_id, name in categories.items()
            if name in TARGET_CLASSES
        }
        new_id_by_source_id = {
            source_name_to_id[name]: index + 1
            for index, name in enumerate(TARGET_CLASSES)
        }

        kept_annotations = []
        for ann in data["annotations"]:
            if ann["image_id"] not in kept_ids:
                continue
            if ann["category_id"] not in new_id_by_source_id:
                continue
            copied = dict(ann)
            copied["category_id"] = new_id_by_source_id[ann["category_id"]]
            kept_annotations.append(copied)

        output_data = {
            **{k: v for k, v in data.items() if k not in {"images", "annotations", "categories"}},
            "images": kept_images,
            "annotations": kept_annotations,
            "categories": [
                {"id": index + 1, "name": name, "supercategory": "damage"}
                for index, name in enumerate(TARGET_CLASSES)
            ],
        }

        with (output_split / "_annotations.coco.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(output_data, handle)

        for image in kept_images:
            source = args.raw / split / image["file_name"]
            destination = output_split / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        totals[split] = len(kept_images)

    print(f"Prepared {sum(totals.values())} images: {totals}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
