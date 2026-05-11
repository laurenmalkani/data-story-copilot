#!/usr/bin/env python3
"""
Extract a journalism-focused subset of DROID 1.0.1 from Hugging Face.

The full DROID dataset is ~40M frame-level rows (~16GB tabular, 512GB with
video). For the workshop story angles in Slide 07 of the deck, we only need
one row per *episode* with the columns that carry editorial signal:

    episode_index, task_category, building, collector_id, date,
    language_instruction[_2,_3], is_episode_successful, task_index

We use DuckDB's httpfs to read the remote parquet files lazily — only the
needed columns and rows are downloaded thanks to predicate pushdown.

Usage:
    pip install duckdb
    python scripts/extract_droid_subset.py

Output:
    droid_episodes.csv  (~76k rows, a few MB — drops directly into the copilot)
"""

import csv
import sys
import time
from pathlib import Path

try:
    import duckdb
except ImportError:
    sys.exit("Missing dependency: pip install duckdb")


HF_BASE = "https://huggingface.co/datasets/lerobot/droid_1.0.1/resolve/main/data/chunk-000"
NUM_FILES = 156  # confirmed via the HF tree API
OUTPUT = Path(__file__).resolve().parent.parent / "droid_episodes.csv"

# Columns that carry editorial / story signal. The rest of the parquet
# (joint positions, camera extrinsics, etc.) is irrelevant for journalism.
#
# DATA-QUALITY NOTES (verified on file-000.parquet, May 2026):
#   - task_category IS IDENTICAL to building. The LeRobot port appears to have
#     aliased these columns; in practice both contain building names like
#     "BWW", "Autolab", "2479 Richard Ct". We drop task_category to avoid
#     misleading the AI prompts. 59 distinct buildings in this file.
#   - date IS IDENTICAL to collector_id (hex strings like "52ca9b6a"). The
#     "date" column is mislabeled — it's an anonymized collector ID, not a
#     calendar date. We drop date and keep collector_id. 71 collectors.
#   - task_index has 816 distinct integer IDs (NOT the "86 task categories"
#     claim from public descriptions). Resolving task names requires the
#     dataset's tasks.json mapping, which we don't load here.
#   - ~80% of episodes have populated language_instruction[_2,_3]. The three
#     columns capture rephrasings of the same instruction, useful for the
#     "language story" angle.
JOURNALISM_COLS = [
    "episode_index",
    "building",                # NOTE: also equals task_category in the parquet
    "collector_id",            # NOTE: also equals date in the parquet
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "is_episode_successful",
    "task_index",
]


def main() -> None:
    print("Connecting to DuckDB...")
    con = duckdb.connect()

    print("Loading httpfs extension (lets DuckDB read parquet over HTTPS)...")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Build the file list — DuckDB will scan these in parallel.
    files = [f"{HF_BASE}/file-{i:03d}.parquet" for i in range(NUM_FILES)]
    files_sql = ", ".join(f"'{u}'" for u in files)

    query = f"""
        COPY (
            SELECT {', '.join(JOURNALISM_COLS)}
            FROM read_parquet([{files_sql}])
            WHERE is_first = true
            ORDER BY collector_id, episode_index
        ) TO '{OUTPUT}' (FORMAT 'csv', HEADER true)
    """

    print(f"Querying {NUM_FILES} parquet files for is_first=true frames...")
    print("(DuckDB reads only the columns + row groups it needs — should be a few MB downloaded, not 16GB.)")
    print("Expect 1-5 minutes depending on bandwidth.\n")

    t0 = time.time()
    con.execute(query)
    elapsed = time.time() - t0

    # Summarize
    with OUTPUT.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"✓ Wrote {row_count:,} episodes to {OUTPUT.name} ({size_mb:.1f} MB) in {elapsed:.0f}s")
    print(f"  Columns: {', '.join(header)}")
    print(f"\nDrop {OUTPUT.name} into data-story-copilot at http://localhost:5174/data-story-copilot/")


if __name__ == "__main__":
    main()
