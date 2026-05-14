"""
Preprocessing script for the batch inference pipeline.

Runs as a SageMaker Processing Job. Queries the database for user
interaction sequences and writes them as JSONL for Batch Transform.

Output: /opt/ml/processing/output/jsonl/requests.jsonl

Usage (inside Processing container — called automatically by SageMaker):
  python preprocess.py \
      --db-connection-string mock://localhost \
      --dataset-name ml-1m \
      --top-k 20 \
      --limit 100
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/opt/ml/processing/output/jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query DB and write JSONL for batch inference."
    )
    parser.add_argument(
        "--db-connection-string",
        required=True,
        dest="db_connection_string",
        help="Database connection string (e.g. mock://localhost, postgresql://...).",
    )
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="Dataset identifier (determines max_item_id for mock backend).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Number of recommendations to request per user.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of users to process (None = all).",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=1,
        dest="num_splits",
        help="Number of JSONL files to split output into. 1 = single file.",
    )
    return parser.parse_args()


def fetch_user_sequences(args):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO: Replace this function with your own data source.        │
    │                                                                │
    │  Currently uses a mock backend that generates random data.     │
    │  To query a real database instead:                             │
    │                                                                │
    │  1. Add a new backend class in db_backend.py                   │
    │     (see the PostgreSQL example in that file), OR              │
    │                                                                │
    │  2. Replace this function body entirely, e.g.:                 │
    │                                                                │
    │     import psycopg2                                            │
    │     conn = psycopg2.connect(args.db_connection_string)         │
    │     cur = conn.cursor()                                        │
    │     cur.execute(                                               │
    │         "SELECT user_id, sequence_item_ids "                   │
    │         "FROM user_interactions LIMIT %s", (args.limit,)       │
    │     )                                                          │
    │     return [                                                   │
    │         {"user_id": row[0],                                    │
    │          "sequence": [int(x) for x in row[1].split(",")]}     │
    │         for row in cur.fetchall()                              │
    │     ]                                                          │
    │                                                                │
    │  The function must return a list of dicts, each with:          │
    │    - "user_id": int                                            │
    │    - "sequence": list[int]  (item IDs, chronological order)    │
    └─────────────────────────────────────────────────────────────────┘
    """
    from db_backend import get_backend

    backend = get_backend(args.db_connection_string)
    try:
        return backend.fetch_user_sequences(
            dataset_name=args.dataset_name, limit=args.limit
        )
    finally:
        backend.close()


def main():
    args = parse_args()

    users = fetch_user_sequences(args)
    logger.info(f"Fetched {len(users)} user sequences")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []
    for user in users:
        records.append(json.dumps({
            "user_id": user["user_id"],
            "sequence": user["sequence"],
            "top_k": args.top_k,
        }))

    import math

    n = max(args.num_splits, 1)
    chunk_size = math.ceil(len(records) / n)
    chunks = [
        records[i:i + chunk_size]
        for i in range(0, len(records), chunk_size)
    ]

    for idx, chunk in enumerate(chunks):
        filename = f"requests_{idx:04d}.jsonl"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            f.write("\n".join(chunk) + "\n")
        logger.info(f"Wrote {len(chunk)} records to {filename}")

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
    )
    logger.info(f"Total: {len(records)} records in {len(chunks)} file(s), {total_size:,} bytes")


if __name__ == "__main__":
    main()
