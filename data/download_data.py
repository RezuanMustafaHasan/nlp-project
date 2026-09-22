"""Download and snapshot the fixed Hugging Face BanglaPRCorpus dataset.

Pins repository: mehedihasanbijoy/BanglaPRCorpus
Revision: 113ab94ee3d1b29a6f504cef00cd65594062b5e8
Files: train.csv, validation.csv, test.csv
Expected columns: source, target, nopr
"""

from __future__ import annotations
import csv
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = 'mehedihasanbijoy/BanglaPRCorpus'
REVISION = '113ab94ee3d1b29a6f504cef00cd65594062b5e8'
FILES = {
    'train': 'train.csv',
    'validation': 'validation.csv',
    'test': 'test.csv',
}
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
EXPECTED_COLUMNS = {'source', 'target', 'nopr'}


def download_csv(split_name: str, filename: str, output_dir: Path) -> Path:
    """Downloads a split CSV file directly from Hugging Face if not already present."""
    dest = output_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"File {dest} already exists ({dest.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return dest

    url = f"{BASE_URL}/{filename}"
    logger.info(f"Downloading {split_name} from {url} to {dest}...")
    headers = {"User-Agent": "BanglaPunctuationRestoration/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, dest.open("wb") as out_f:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
    logger.info(f"Downloaded {filename} ({dest.stat().st_size / (1024*1024):.1f} MB).")
    return dest


def verify_and_record_download(
    output_dir: str | Path = "data/raw_hf",
    manifest_path: str | Path = "data/manifests/download.json"
) -> Dict[str, Any]:
    """Downloads all 3 split CSV files and verifies columns and row counts."""
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    row_counts: Dict[str, int] = {}

    for split_name, filename in FILES.items():
        csv_file = download_csv(split_name, filename, output_dir)
        count = 0
        with csv_file.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            assert EXPECTED_COLUMNS.issubset(fieldnames), (
                f"Columns in {filename} ({fieldnames}) missing expected {EXPECTED_COLUMNS}"
            )
            for _ in reader:
                count += 1
        row_counts[split_name] = count
        logger.info(f"Verified {split_name}: {count:,} rows. Columns: {sorted(fieldnames)}")

    manifest_record = {
        'repository': REPO_ID,
        'revision': REVISION,
        'data_files': FILES,
        'expected_columns': list(EXPECTED_COLUMNS),
        'rows': row_counts,
        'total_rows': sum(row_counts.values()),
    }

    manifest_path.write_text(
        json.dumps(manifest_record, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"Saved download manifest to {manifest_path}")
    return manifest_record


if __name__ == "__main__":
    verify_and_record_download()
