#!/usr/bin/env python3
"""
Simple JSON-to-text converter that preserves textual values at any depth.

Usage:
    python json_to_txt.py input.json output.txt
"""

import argparse
import json
from pathlib import Path
from typing import List


def extract_text(data) -> List[str]:
    if isinstance(data, str):
        return [data]
    if isinstance(data, dict):
        texts: List[str] = []
        for value in data.values():
            texts.extend(extract_text(value))
        return texts
    if isinstance(data, list):
        texts: List[str] = []
        for value in data:
            texts.extend(extract_text(value))
        return texts
    return []


def convert(input_path: Path, output_path: Path) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    lines = extract_text(raw)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON to plain text.")
    parser.add_argument("input", type=Path, help="Path to source JSON file.")
    parser.add_argument("output", type=Path, help="Path to target text file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
