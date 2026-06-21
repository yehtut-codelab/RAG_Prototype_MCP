"""Run once to enable pgvector extension in PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

import psycopg
from config import settings  # noqa: E402


def setup():
    print(f"Connecting to: {settings.postgres_url.split('@')[-1]}")
    try:
        with psycopg.connect(settings.postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        print("pgvector extension enabled.")
    except Exception as e:
        print(f"Error enabling pgvector: {e}")
        sys.exit(1)

    print("\nVector store tables will be auto-created on first document ingest.")
    print("Setup complete.")


if __name__ == "__main__":
    setup()
