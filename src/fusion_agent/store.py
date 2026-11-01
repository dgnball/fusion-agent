"""One DuckDB file holds every modality.

Reports and images carry embeddings next to their metadata, so a single SQL query can combine a time window,
a location filter and vector similarity. Telemetry stays as plain rows because the useful questions about it
are analytical (gaps, zone entries, speeds), not semantic.
"""

import json
from datetime import datetime
from pathlib import Path

import duckdb

from .embed import Embedder

SCHEMA = """
CREATE OR REPLACE TABLE reports (id VARCHAR PRIMARY KEY, ts TIMESTAMP, location VARCHAR, author VARCHAR,
    title VARCHAR, body VARCHAR, embedding FLOAT[]);
CREATE OR REPLACE TABLE images (id VARCHAR PRIMARY KEY, ts TIMESTAMP, camera VARCHAR, lat DOUBLE,
    lon DOUBLE, path VARCHAR, embedding FLOAT[]);
CREATE OR REPLACE TABLE vessels (vessel_id VARCHAR PRIMARY KEY, name VARCHAR, type VARCHAR);
CREATE OR REPLACE TABLE zones (name VARCHAR, min_lat DOUBLE, max_lat DOUBLE, min_lon DOUBLE, max_lon DOUBLE);
CREATE OR REPLACE TABLE areas (name VARCHAR, min_lat DOUBLE, max_lat DOUBLE, min_lon DOUBLE, max_lon DOUBLE);
CREATE OR REPLACE TABLE cameras (camera VARCHAR, name VARCHAR, lat DOUBLE, lon DOUBLE);
CREATE OR REPLACE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR);
"""


def ingest(data_dir: Path, db_path: Path, embedder: Embedder) -> dict:
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)

    reports = [json.loads(line) for line in (data_dir / "reports.jsonl").read_text().splitlines()]
    vectors = embedder.text([f"{r['title']}. {r['body']}" for r in reports])
    con.executemany(
        "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            [r["id"], r["ts"], r["location"], r["author"], r["title"], r["body"], v.tolist()]
            for r, v in zip(reports, vectors, strict=True)
        ],
    )

    images = [json.loads(line) for line in (data_dir / "images.jsonl").read_text().splitlines()]
    vectors = embedder.images([data_dir / i["path"] for i in images])
    con.executemany(
        "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            [i["id"], i["ts"], i["camera"], i["lat"], i["lon"], str(data_dir / i["path"]), v.tolist()]
            for i, v in zip(images, vectors, strict=True)
        ],
    )

    con.execute(
        "CREATE OR REPLACE TABLE tracks AS SELECT vessel_id, ts::TIMESTAMP AS ts, lat, lon, speed_kn, "
        "heading FROM read_csv(?) ORDER BY vessel_id, ts",
        [str(data_dir / "tracks.csv")],
    )
    con.executemany(
        "INSERT INTO vessels VALUES (?, ?, ?)",
        [[v["vessel_id"], v["name"], v["type"]] for v in json.loads((data_dir / "vessels.json").read_text())],
    )
    con.executemany(
        "INSERT INTO zones VALUES (?, ?, ?, ?, ?)",
        [
            [z["name"], z["min_lat"], z["max_lat"], z["min_lon"], z["max_lon"]]
            for z in json.loads((data_dir / "zones.json").read_text())
        ],
    )
    con.executemany(
        "INSERT INTO areas VALUES (?, ?, ?, ?, ?)",
        [
            [a["name"], a["min_lat"], a["max_lat"], a["min_lon"], a["max_lon"]]
            for a in json.loads((data_dir / "areas.json").read_text())
        ],
    )
    con.executemany(
        "INSERT INTO cameras VALUES (?, ?, ?, ?)",
        [
            [key, c["name"], c["lat"], c["lon"]]
            for key, c in json.loads((data_dir / "cameras.json").read_text()).items()
        ],
    )
    con.execute("INSERT INTO meta VALUES ('embedder', ?)", [embedder.name])
    counts = {
        table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("reports", "images", "tracks", "vessels")
    }
    con.close()
    return counts


class Store:
    def __init__(self, db_path: Path, embedder: Embedder):
        self.con = duckdb.connect(str(db_path), read_only=True)
        stored = self.con.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()[0]
        if stored != embedder.name:
            raise ValueError(f"Database was built with the {stored} embedder, not {embedder.name}")
        self.embedder = embedder

    def rows(self, sql: str, params: list | None = None) -> list[dict]:
        cursor = self.con.execute(sql, params or [])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def coverage(self) -> dict:
        start, end = self.con.execute(
            "SELECT least((SELECT min(ts) FROM tracks), (SELECT min(ts) FROM reports)), "
            "greatest((SELECT max(ts) FROM tracks), (SELECT max(ts) FROM reports))"
        ).fetchone()
        return {
            "start": start,
            "end": end,
            "zones": self.rows("SELECT * FROM zones"),
            "areas": self.rows("SELECT * FROM areas"),
            "cameras": self.rows("SELECT * FROM cameras"),
        }

    @staticmethod
    def _window(column: str, start: datetime | None, end: datetime | None) -> tuple[str, list]:
        clauses, params = [], []
        if start:
            clauses.append(f"{column} >= ?")
            params.append(start)
        if end:
            clauses.append(f"{column} <= ?")
            params.append(end)
        return (" AND ".join(clauses) or "TRUE"), params

    def search_reports(self, query, start=None, end=None, limit=5):
        where, params = self._window("ts", start, end)
        vector = self.embedder.text([query])[0].tolist()
        return self.rows(
            f"SELECT id, ts, location, author, title, body, "
            f"list_cosine_similarity(embedding, ?::FLOAT[]) AS score FROM reports WHERE {where} "
            f"ORDER BY score DESC LIMIT ?",
            [vector, *params, limit],
        )

    def search_images(self, query, camera=None, start=None, end=None, limit=5):
        where, params = self._window("ts", start, end)
        if camera:
            where += " AND camera = ?"
            params.append(camera)
        vector = self.embedder.image_query([query])[0].tolist()
        return self.rows(
            f"SELECT id, ts, camera, lat, lon, list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
            f"FROM images WHERE {where} ORDER BY score DESC LIMIT ?",
            [vector, *params, limit],
        )

    def vessels(self):
        return self.rows("SELECT * FROM vessels ORDER BY vessel_id")

    def resolve_vessel(self, vessel: str) -> dict | None:
        found = self.rows(
            "SELECT * FROM vessels WHERE lower(vessel_id) = lower(?) OR lower(name) = lower(?)",
            [vessel, vessel],
        )
        return found[0] if found else None

    def track(self, vessel_id, start=None, end=None):
        """Position reports, each labelled with the restricted zone or named area it falls in."""
        where, params = self._window("t.ts", start, end)
        return self.rows(
            f"""
            SELECT t.*, coalesce(
                (SELECT name FROM zones z WHERE t.lat BETWEEN z.min_lat AND z.max_lat
                    AND t.lon BETWEEN z.min_lon AND z.max_lon LIMIT 1),
                (SELECT name FROM areas a WHERE t.lat BETWEEN a.min_lat AND a.max_lat
                    AND t.lon BETWEEN a.min_lon AND a.max_lon LIMIT 1),
                'Open water') AS area
            FROM tracks t WHERE t.vessel_id = ? AND {where} ORDER BY t.ts""",
            [vessel_id, *params],
        )

    def zone_entries(self, start=None, end=None):
        """Group consecutive position reports inside each zone into visits."""
        where, params = self._window("t.ts", start, end)
        return self.rows(
            f"""
            WITH inside AS (
                SELECT t.*, z.name AS zone, v.name AS vessel_name,
                    CASE WHEN t.ts - lag(t.ts) OVER (PARTITION BY t.vessel_id, z.name ORDER BY t.ts)
                         <= INTERVAL 15 MINUTES THEN 0 ELSE 1 END AS new_visit
                FROM tracks t JOIN zones z
                  ON t.lat BETWEEN z.min_lat AND z.max_lat AND t.lon BETWEEN z.min_lon AND z.max_lon
                JOIN vessels v USING (vessel_id)
                WHERE {where}),
            numbered AS (SELECT *, sum(new_visit) OVER (PARTITION BY vessel_id, zone ORDER BY ts) AS visit
                         FROM inside)
            SELECT vessel_id, vessel_name, zone, min(ts) AS first_seen, max(ts) AS last_seen,
                   count(*) AS reports, round(avg(speed_kn), 1) AS mean_speed_kn
            FROM numbered GROUP BY vessel_id, vessel_name, zone, visit ORDER BY first_seen""",
            params,
        )

    def ais_gaps(self, start=None, end=None, minutes=30):
        where, params = self._window("ts", start, end)
        return self.rows(
            f"""
            WITH ordered AS (
                SELECT vessel_id, ts, lat, lon,
                    lag(ts) OVER w AS prev_ts, lag(lat) OVER w AS prev_lat, lag(lon) OVER w AS prev_lon
                FROM tracks WINDOW w AS (PARTITION BY vessel_id ORDER BY ts))
            SELECT o.vessel_id, v.name AS vessel_name, prev_ts AS last_seen, ts AS seen_again,
                   date_diff('minute', prev_ts, ts) AS gap_minutes,
                   prev_lat AS last_lat, prev_lon AS last_lon, lat AS next_lat, lon AS next_lon
            FROM ordered o JOIN vessels v USING (vessel_id)
            WHERE prev_ts IS NOT NULL AND ts - prev_ts > INTERVAL (?) MINUTES AND {where}
            ORDER BY last_seen""",
            [minutes, *params],
        )

    def report(self, report_id):
        found = self.rows("SELECT id, ts, location, author, title, body FROM reports WHERE id = ?", [report_id])
        return found[0] if found else None

    def image(self, image_id):
        found = self.rows("SELECT id, ts, camera, lat, lon, path FROM images WHERE id = ?", [image_id])
        return found[0] if found else None
