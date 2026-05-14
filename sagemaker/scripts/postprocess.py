"""
Postprocessing script for the batch inference pipeline.

Runs as a SageMaker Processing Job. Reads Batch Transform output
and writes recommendations back to the database.

Input:  /opt/ml/processing/input/transform_output/*.out
Output: /opt/ml/processing/output/results/summary.json

Usage (inside Processing container — called automatically by SageMaker):
  python postprocess.py \
      --db-connection-string mock://localhost
"""

import argparse
import glob
import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "/opt/ml/processing/input/transform_output"
OUTPUT_DIR = "/opt/ml/processing/output/results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read Batch Transform output and write recommendations to DB."
    )
    parser.add_argument(
        "--db-connection-string",
        required=True,
        dest="db_connection_string",
        help="Database connection string (e.g. mock://localhost, postgresql://...).",
    )
    return parser.parse_args()


def write_recommendations(args, recommendations):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO: Replace this function with your own data sink.          │
    │                                                                │
    │  Currently uses a mock backend that logs results to stdout.    │
    │  To write to a real database instead:                          │
    │                                                                │
    │  1. Add a new backend class in db_backend.py                   │
    │     (see the PostgreSQL example in that file), OR              │
    │                                                                │
    │  2. Replace this function body entirely, e.g.:                 │
    │                                                                │
    │     import psycopg2                                            │
    │     conn = psycopg2.connect(args.db_connection_string)         │
    │     cur = conn.cursor()                                        │
    │     for rec in recommendations:                                │
    │         cur.execute(                                           │
    │             "INSERT INTO recommendations "                     │
    │             "(user_id, item_ids) VALUES (%s, %s) "             │
    │             "ON CONFLICT (user_id) DO UPDATE "                 │
    │             "SET item_ids = EXCLUDED.item_ids",                │
    │             (rec["user_id"],                                   │
    │              ",".join(str(x) for x in rec["item_ids"])),       │
    │         )                                                      │
    │     conn.commit()                                              │
    │                                                                │
    │  Each dict in `recommendations` has:                           │
    │    - "user_id": int                                            │
    │    - "item_ids": list[int]  (recommended items, ranked)        │
    └─────────────────────────────────────────────────────────────────┘
    """
    if not recommendations:
        logger.warning("No recommendation records found — nothing to write.")
        return

    from db_backend import get_backend

    backend = get_backend(args.db_connection_string)
    try:
        backend.write_recommendations(recommendations)
    finally:
        backend.close()


def main():
    args = parse_args()

    recommendations = []
    for filepath in sorted(glob.glob(os.path.join(INPUT_DIR, "**"), recursive=True)):
        if not os.path.isfile(filepath):
            continue
        logger.info(f"Reading: {filepath}")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    recommendations.append(json.loads(line))

    logger.info(f"Parsed {len(recommendations)} recommendation records")

    write_recommendations(args, recommendations)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "total_users": len(recommendations),
        "avg_recommendations": round(
            sum(len(r.get("item_ids", [])) for r in recommendations)
            / max(len(recommendations), 1),
            1,
        ),
        "sample_results": recommendations[:3],
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary['total_users']} users, "
                f"avg {summary['avg_recommendations']} recommendations each")


if __name__ == "__main__":
    main()
