"""Ingest Chinook DB rows into the pgvector knowledge base for semantic RAG search.

Usage:
    python ingest_chinook.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

import psycopg
from langchain_core.documents import Document
from config import settings
from vector_store import add_documents

CHINOOK_QUERIES = {
    "tracks": """
        SELECT
            t.track_id,
            t.name            AS track,
            ar.name           AS artist,
            al.title          AS album,
            g.name            AS genre,
            mt.name           AS media_type,
            t.composer,
            t.milliseconds,
            t.unit_price
        FROM track t
        JOIN album      al ON t.album_id      = al.album_id
        JOIN artist     ar ON al.artist_id    = ar.artist_id
        JOIN genre       g ON t.genre_id      = g.genre_id
        JOIN media_type mt ON t.media_type_id = mt.media_type_id
    """,
    "customers": """
        SELECT
            c.customer_id,
            c.first_name || ' ' || c.last_name  AS customer,
            c.company,
            c.city,
            c.country,
            c.email,
            e.first_name || ' ' || e.last_name  AS support_rep
        FROM customer c
        LEFT JOIN employee e ON c.support_rep_id = e.employee_id
    """,
    "invoices": """
        SELECT
            i.invoice_id,
            c.first_name || ' ' || c.last_name AS customer,
            c.country,
            i.invoice_date,
            i.total,
            i.billing_city,
            i.billing_country
        FROM invoice i
        JOIN customer c ON i.customer_id = c.customer_id
    """,
    "playlists": """
        SELECT
            p.name      AS playlist,
            t.name      AS track,
            ar.name     AS artist,
            al.title    AS album
        FROM playlist p
        JOIN playlist_track pt ON p.playlist_id = pt.playlist_id
        JOIN track          t  ON pt.track_id   = t.track_id
        JOIN album         al  ON t.album_id    = al.album_id
        JOIN artist        ar  ON al.artist_id  = ar.artist_id
    """,
}

PRIMARY_TABLE = {
    "tracks": "track",
    "customers": "customer",
    "invoices": "invoice",
    "playlists": "playlist",
}


def row_to_text(table: str, row: dict) -> str:
    pairs = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
    return f"[{table}] {pairs}"


def ingest_table(cur, table: str, sql: str, batch_size: int = 200) -> int:
    cur.execute(sql)
    cols = [desc.name for desc in cur.description]
    total = 0
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        docs = []
        for row in rows:
            record = dict(zip(cols, row))
            text = row_to_text(table, record)
            pk_col = next((c for c in cols if c.endswith("_id")), None)
            metadata = {
                "source": f"chinook:{table}",
                "table": table,
                **({"id": str(record[pk_col])} if pk_col else {}),
            }
            docs.append(Document(page_content=text, metadata=metadata))
        add_documents(docs)
        total += len(docs)
        print(f"  {table}: {total} rows ingested...")
    return total


def main():
    print(f"Connecting to {settings.postgres_url.split('@')[-1]}\n")
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            existing = {r[0] for r in cur.fetchall()}

            chinook_tables = {"track", "customer", "invoice", "playlist",
                              "album", "artist", "genre", "employee"}
            found = existing & chinook_tables
            if not found:
                print("No Chinook tables found. Make sure Chinook data is loaded.")
                sys.exit(1)

            print(f"Chinook tables detected: {sorted(found)}\n")

            total = 0
            for key, sql in CHINOOK_QUERIES.items():
                if PRIMARY_TABLE[key] not in existing:
                    print(f"Skipping '{key}' — table not found.\n")
                    continue
                print(f"Ingesting {key}...")
                n = ingest_table(cur, key, sql.strip())
                print(f"  Done: {n} rows\n")
                total += n

    print(f"Chinook ingest complete. Total rows stored: {total}")


if __name__ == "__main__":
    main()
