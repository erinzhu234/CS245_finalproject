"""
Lightweight dataset processor to produce the JSONL files expected by WebSocietySimulator.

Usage (Yelp):
    python data_process.py --input /path/to/raw_yelp --output /path/to/processed_yelp

Input directory must contain:
    - yelp_academic_dataset_business.json
    - yelp_academic_dataset_user.json
    - yelp_academic_dataset_review.json

Output directory will contain:
    - item.json     (businesses)
    - user.json
    - review.json   (reviews)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, TextIO


def _open_text(path: Path) -> TextIO:
    """Open a JSON or JSON.GZ file for reading text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_json_lines(path: Path) -> Iterator[dict]:
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_yelp(input_dir: Path, output_dir: Path) -> None:
    """Convert Yelp academic JSON to simulator-friendly JSONL files."""
    biz_raw = input_dir / "yelp_academic_dataset_business.json"
    user_raw = input_dir / "yelp_academic_dataset_user.json"
    review_raw = input_dir / "yelp_academic_dataset_review.json"

    for p in [biz_raw, user_raw, review_raw]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    item_out = output_dir / "item.json"
    user_out = output_dir / "user.json"
    review_out = output_dir / "review.json"

    print(f"[yelp] writing {item_out} ...")
    def _iter_items() -> Iterator[dict]:
        for biz in _iter_json_lines(biz_raw):
            categories = biz.get("categories")
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]
            elif not isinstance(categories, list):
                categories = []
            yield {
                "item_id": biz.get("business_id"),
                "name": biz.get("name"),
                "address": biz.get("address"),
                "city": biz.get("city"),
                "state": biz.get("state"),
                "postal_code": biz.get("postal_code"),
                "stars": biz.get("stars"),
                "review_count": biz.get("review_count"),
                "categories": categories,
                "attributes": biz.get("attributes"),
                "source": "yelp",
            }
    _write_jsonl(item_out, _iter_items())

    print(f"[yelp] writing {user_out} ...")
    def _iter_users() -> Iterator[dict]:
        for user in _iter_json_lines(user_raw):
            yield {
                "user_id": user.get("user_id"),
                "name": user.get("name"),
                "review_count": user.get("review_count"),
                "yelping_since": user.get("yelping_since"),
                "useful": user.get("useful"),
                "funny": user.get("funny"),
                "cool": user.get("cool"),
                "elite": user.get("elite"),
                "friends": user.get("friends"),
                "fans": user.get("fans"),
                "average_stars": user.get("average_stars"),
                "compliment_list": {k: v for k, v in user.items() if k.startswith("compliment_")},
                "source": "yelp",
            }
    _write_jsonl(user_out, _iter_users())

    print(f"[yelp] writing {review_out} ...")
    def _iter_reviews() -> Iterator[dict]:
        for rev in _iter_json_lines(review_raw):
            yield {
                "review_id": rev.get("review_id"),
                "user_id": rev.get("user_id"),
                "item_id": rev.get("business_id"),
                "stars": rev.get("stars"),
                "text": rev.get("text"),
                "date": rev.get("date"),
                "useful": rev.get("useful"),
                "funny": rev.get("funny"),
                "cool": rev.get("cool"),
                "source": "yelp",
            }
    _write_jsonl(review_out, _iter_reviews())

    print(f"Done. Files written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process raw datasets into WebSocietySimulator format.")
    parser.add_argument("--input", required=True, help="Directory with raw Yelp files.")
    parser.add_argument("--output", required=True, help="Directory to write processed JSONL files.")
    parser.add_argument(
        "--dataset",
        default="yelp",
        choices=["yelp"],
        help="Dataset to process (currently only Yelp supported).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    if args.dataset == "yelp":
        process_yelp(input_dir, output_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
