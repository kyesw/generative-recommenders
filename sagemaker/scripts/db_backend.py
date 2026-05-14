"""
Database backend abstraction for batch pipeline pre/post processing.

Usage:
    backend = get_backend("mock://localhost")
    users = backend.fetch_user_sequences(dataset_name="ml-1m", limit=100)
    backend.write_recommendations([{"user_id": 1, "item_ids": [10, 20]}])

To add a real backend:
    1. Subclass DatabaseBackend
    2. Implement fetch_user_sequences() and write_recommendations()
    3. Register it in get_backend() with your URI scheme

Example — PostgreSQL:
    class PostgresDatabaseBackend(DatabaseBackend):
        def __init__(self, connection_string: str):
            import psycopg2
            self._conn = psycopg2.connect(connection_string)

        def fetch_user_sequences(self, dataset_name, limit=None):
            cur = self._conn.cursor()
            query = "SELECT user_id, sequence_item_ids FROM user_interactions"
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
            return [
                {"user_id": row[0], "sequence": [int(x) for x in row[1].split(",")]}
                for row in cur.fetchall()
            ]

        def write_recommendations(self, recommendations):
            cur = self._conn.cursor()
            for rec in recommendations:
                cur.execute(
                    "INSERT INTO recommendations (user_id, item_ids) VALUES (%s, %s) "
                    "ON CONFLICT (user_id) DO UPDATE SET item_ids = EXCLUDED.item_ids",
                    (rec["user_id"], ",".join(str(x) for x in rec["item_ids"])),
                )
            self._conn.commit()

        def close(self):
            self._conn.close()

    Then in get_backend():
        if scheme in ("postgresql", "postgres"):
            return PostgresDatabaseBackend(connection_string)
"""

import abc
import logging

logger = logging.getLogger(__name__)


class DatabaseBackend(abc.ABC):
    """Abstract interface for reading user sequences and writing recommendations."""

    @abc.abstractmethod
    def fetch_user_sequences(self, dataset_name: str, limit: int = None) -> list:
        """
        Return user interaction sequences for batch inference.

        Each dict must have:
            user_id: int
            sequence: list[int]  — item IDs in chronological order
        """
        ...

    @abc.abstractmethod
    def write_recommendations(self, recommendations: list) -> None:
        """
        Write recommendation results back to the database.

        Each dict has:
            user_id: int
            item_ids: list[int]  — recommended item IDs, ranked
        """
        ...

    def close(self) -> None:
        pass


class MockDatabaseBackend(DatabaseBackend):
    """
    Mock backend that generates synthetic user sequences for testing.

    Generates N users with random-length sequences of item IDs.
    Writes recommendations to the log.
    """

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._written = []

    def fetch_user_sequences(self, dataset_name: str, limit: int = None) -> list:
        import random

        random.seed(42)
        num_users = limit or 100
        max_item_id = {"ml-1m": 3952, "ml-20m": 131262, "amzn-books": 367982}.get(
            dataset_name, 1000
        )

        users = []
        for uid in range(1, num_users + 1):
            seq_len = random.randint(5, 50)
            sequence = [random.randint(1, max_item_id) for _ in range(seq_len)]
            users.append({"user_id": uid, "sequence": sequence})

        logger.info(f"MockDB: generated {len(users)} synthetic user sequences")
        return users

    def write_recommendations(self, recommendations: list) -> None:
        self._written.extend(recommendations)
        logger.info(
            f"MockDB: received {len(recommendations)} recommendation records "
            f"(total stored: {len(self._written)})"
        )
        for rec in recommendations[:3]:
            logger.info(f"  sample: user_id={rec['user_id']}, items={rec['item_ids'][:5]}...")


def get_backend(connection_string: str) -> DatabaseBackend:
    """
    Factory: create a DatabaseBackend from a connection string.

    Supported schemes:
        mock://       -> MockDatabaseBackend (synthetic data for testing)
        # postgresql:// -> PostgresDatabaseBackend (uncomment when ready)
        # dynamodb://   -> DynamoDBDatabaseBackend (uncomment when ready)
    """
    scheme = connection_string.split("://")[0].lower()

    if scheme == "mock":
        return MockDatabaseBackend(connection_string)

    raise ValueError(
        f"Unsupported database scheme '{scheme}' in '{connection_string}'. "
        f"Supported: mock://"
    )
