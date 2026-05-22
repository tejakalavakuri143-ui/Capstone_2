import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "invoices.db"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():

    conn = _connect()

    cur = conn.cursor()

    cur.execute("""

    CREATE TABLE IF NOT EXISTS invoice_reports (

        invoice_id TEXT PRIMARY KEY,

        report_json TEXT

    )

    """)

    cur.execute("""

    CREATE TABLE IF NOT EXISTS processed_files (

        file_hash TEXT PRIMARY KEY,

        file_path TEXT NOT NULL,

        file_name TEXT NOT NULL,

        processed_at TEXT NOT NULL,

        invoice_id TEXT,

        status TEXT,

        error TEXT

    )

    """)

    cur.execute("""

    CREATE TABLE IF NOT EXISTS human_reviews (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        invoice_id TEXT NOT NULL,

        decision TEXT NOT NULL,

        comments TEXT,

        created_at TEXT NOT NULL

    )

    """)

    conn.commit()

    conn.close()


def save_report(report: dict):

    conn = _connect()

    cur = conn.cursor()

    invoice_id = report[
        "report_metadata"
    ]["invoice_id"]

    cur.execute("""

    INSERT OR REPLACE INTO invoice_reports

    VALUES (?, ?)

    """, (

        invoice_id,

        json.dumps(report)

    ))

    conn.commit()

    conn.close()


def load_reports():

    conn = _connect()

    cur = conn.cursor()

    cur.execute("""

    SELECT report_json
    FROM invoice_reports

    """)

    rows = cur.fetchall()

    conn.close()

    reports_by_id = {}

    for row in rows:
        report = json.loads(row[0])
        invoice_id = report.get("report_metadata", {}).get("invoice_id")
        if invoice_id:
            reports_by_id[invoice_id] = report

    if REPORTS_DIR.exists():
        for path in REPORTS_DIR.glob("*_report.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            invoice_id = report.get("report_metadata", {}).get("invoice_id")
            if not invoice_id:
                continue

            current = reports_by_id.get(invoice_id)
            if not current:
                reports_by_id[invoice_id] = report
                continue

            current_has_human_review = bool(current.get("human_review"))
            json_has_more_line_items = len(report.get("line_items") or []) > len(
                current.get("line_items") or []
            )
            json_is_newer = (
                report.get("report_metadata", {}).get("generated_at", "")
                >= current.get("report_metadata", {}).get("generated_at", "")
            )

            if not current_has_human_review and (
                json_has_more_line_items or json_is_newer
            ):
                reports_by_id[invoice_id] = report

    return sorted(
        reports_by_id.values(),
        key=lambda report: report.get("report_metadata", {}).get("generated_at", ""),
        reverse=True,
    )
def update_report_status(
    invoice_id: str,
    decision: str,
    comments: str = "",
    decided_at: str | None = None,
) -> bool:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT report_json FROM invoice_reports WHERE invoice_id = ? LIMIT 1",
        (invoice_id,),
    )
    row = cur.fetchone()

    if not row:
        conn.close()
        return False

    report = json.loads(row[0])
    normalized = "Approve" if str(decision).lower().startswith("approve") else "Reject"
    report["human_review"] = {
        "decision": normalized,
        "comments": comments,
        "decided_at": decided_at,
    }
    report["recommendation"] = normalized
    report.setdefault("validation_result", {})["status"] = (
        "HUMAN_APPROVED" if normalized == "Approve" else "HUMAN_REJECTED"
    )

    cur.execute(
        "UPDATE invoice_reports SET report_json = ? WHERE invoice_id = ?",
        (json.dumps(report), invoice_id),
    )
    cur.execute(
        "UPDATE processed_files SET status = ? WHERE invoice_id = ?",
        (report["validation_result"]["status"], invoice_id),
    )
    conn.commit()
    conn.close()
    return True


def clear_reports():

    conn = _connect()

    cur = conn.cursor()

    cur.execute("""

    DELETE FROM invoice_reports

    """)

    conn.commit()

    conn.close()

    print(
        "All invoice reports deleted."
    )


def mark_file_processed(
    file_hash: str,
    file_path: str,
    file_name: str,
    processed_at: str,
    invoice_id: str | None = None,
    status: str | None = None,
    error: str | None = None,
):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""

    INSERT OR REPLACE INTO processed_files
    (file_hash, file_path, file_name, processed_at, invoice_id, status, error)
    VALUES (?, ?, ?, ?, ?, ?, ?)

    """, (
        file_hash,
        file_path,
        file_name,
        processed_at,
        invoice_id,
        status,
        error,
    ))
    conn.commit()
    conn.close()


def is_file_processed(file_hash: str) -> bool:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM processed_files WHERE file_hash = ? LIMIT 1",
        (file_hash,),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def load_processed_files():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""

    SELECT file_hash, file_path, file_name, processed_at, invoice_id, status, error
    FROM processed_files
    ORDER BY processed_at DESC

    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "file_hash": r[0],
            "file_path": r[1],
            "file_name": r[2],
            "processed_at": r[3],
            "invoice_id": r[4],
            "status": r[5],
            "error": r[6],
        }
        for r in rows
    ]


def save_human_review(invoice_id: str, decision: str, comments: str, created_at: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""

    INSERT INTO human_reviews (invoice_id, decision, comments, created_at)
    VALUES (?, ?, ?, ?)

    """, (invoice_id, decision, comments, created_at))
    conn.commit()
    conn.close()


def load_human_reviews():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""

    SELECT invoice_id, decision, comments, created_at
    FROM human_reviews
    ORDER BY created_at DESC

    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "invoice_id": r[0],
            "decision": r[1],
            "comments": r[2],
            "timestamp": r[3],
        }
        for r in rows
    ]
init_db()
