import sys
sys.path.insert(0, 'src')
from database.postgres import get_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

engine = get_engine()
with Session(engine) as s:
    # INFY remaining
    r = s.execute(text(
        "SELECT id, file_name FROM documents "
        "WHERE company_id=82 AND embedded=FALSE ORDER BY id"
    )).fetchall()
    print(f"INFY not-yet-embedded ({len(r)} docs):")
    for row in r:
        print(f"  doc_id={row[0]}  {row[1]}")

    # check last processing_log for INFY
    last = s.execute(text(
        "SELECT pl.stage, pl.status, pl.message, pl.created_at "
        "FROM processing_log pl "
        "JOIN documents d ON d.id = pl.document_id "
        "WHERE d.company_id = 82 "
        "ORDER BY pl.created_at DESC LIMIT 3"
    )).fetchall()
    print("\nINFY last log entries:")
    for row in last:
        print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
