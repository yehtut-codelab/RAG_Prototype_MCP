import psycopg
from collections import defaultdict


def load_schema(postgres_url: str) -> str:
    conn_url = postgres_url
    if conn_url.startswith("postgresql+psycopg://"):
        conn_url = conn_url.replace("postgresql+psycopg://", "postgresql://", 1)

    with psycopg.connect(conn_url) as conn:
        with conn.cursor() as cur:

            cur.execute("""
                SELECT c.table_name, c.column_name, c.data_type,
                       c.is_nullable, c.column_default
                FROM information_schema.tables t
                JOIN information_schema.columns c
                  ON t.table_name = c.table_name AND t.table_schema = c.table_schema
                WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                ORDER BY c.table_name, c.ordinal_position
            """)
            col_rows = cur.fetchall()

            cur.execute("""
                SELECT tc.table_name, kcu.column_name,
                       ccu.table_name AS ref_table, ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                 AND tc.table_schema = ccu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
            """)
            fk_rows = cur.fetchall()

            cur.execute("""
                SELECT kcu.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = 'public'
            """)
            pk_rows = cur.fetchall()

    pks = defaultdict(set)
    for table, col in pk_rows:
        pks[table].add(col)

    fks = defaultdict(dict)
    for table, col, ref_table, ref_col in fk_rows:
        fks[table][col] = (ref_table, ref_col)

    tables = defaultdict(list)
    for table, col, dtype, nullable, default in col_rows:
        tables[table].append((col, dtype, nullable, default))

    lines = ["DATABASE SCHEMA", "=" * 40]
    for table in sorted(tables):
        lines.append(f"\nTABLE: {table}")
        for col, dtype, nullable, default in tables[table]:
            tags = []
            if col in pks[table]:
                tags.append("PK")
            if col in fks[table]:
                ref_t, ref_c = fks[table][col]
                tags.append(f"FK → {ref_t}.{ref_c}")
            if nullable == "NO":
                tags.append("NOT NULL")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {col} ({dtype}){tag_str}")

    lines.append("\n" + "=" * 40)
    return "\n".join(lines)
