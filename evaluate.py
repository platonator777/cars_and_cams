"""Передает результаты в неизмененный официальный скрипт оценки."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    """Разбирает аргументы и вызывает официальный оценщик с теми же файлами."""
    parser = argparse.ArgumentParser(description="Call the byte-identical supplied official evaluator")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--embeddings")
    parser.add_argument("--query")
    parser.add_argument("--gallery")
    parser.add_argument("--json")
    args = parser.parse_args()
    command = [sys.executable, str(HERE / "official/evaluate.py"), "--gt", args.gt, "--submission", args.submission]
    for flag in ("candidates", "embeddings", "query", "gallery", "json"):
        value = getattr(args, flag)
        if value:
            command += ["--" + flag, value]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
