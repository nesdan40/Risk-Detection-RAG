import sys
sys.path.insert(0, 'src')
from database.postgres import get_engine
from sqlalchemy import text
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient

DEMO_TICKERS = ['DHFL', 'HDFC', 'INFY', 'JPINFRATEC', 'SPICEJET', 'TATASTEEL', 'TCS']

engine = get_engine()
with Session(engine) as s:
    rows = s.execute(text(
        "SELECT c.ticker, c.id,"
        " (SELECT COUNT(*) FROM documents d WHERE d.company_id = c.id) as docs,"
        " (SELECT COUNT(*) FROM documents d WHERE d.company_id = c.id AND d.text_extracted) as extracted,"
        " (SELECT COUNT(*) FROM documents d WHERE d.company_id = c.id AND d.chunked) as chunked,"
        " (SELECT COUNT(*) FROM documents d WHERE d.company_id = c.id AND d.embedded) as embedded,"
        " (SELECT COUNT(*) FROM shareholding_patterns sp WHERE sp.company_id = c.id) as shareholding,"
        " (SELECT COUNT(*) FROM extracted_features ef WHERE ef.company_id = c.id) as features,"
        " (SELECT COUNT(*) FROM predictions p WHERE p.company_id = c.id) as predictions"
        " FROM companies c"
        " WHERE c.ticker IN ('DHFL','HDFC','INFY','JPINFRATEC','SPICEJET','TATASTEEL','TCS')"
        " ORDER BY c.ticker"
    )).fetchall()

    print('Ticker       ID   Docs Extr Chkd Embd Shrh Feat Pred')
    print('-' * 65)
    found = []
    for r in rows:
        found.append(r[0])
        print(f'{r[0]:<12} {r[1]:>4} {r[2]:>5} {r[3]:>5} {r[4]:>5} {r[5]:>5} {r[6]:>5} {r[7]:>5} {r[8]:>5}')

    missing = [t for t in DEMO_TICKERS if t not in found]
    if missing:
        print('NOT IN DB: ' + str(missing))
    else:
        print('All 7 demo companies present in DB.')

qc = QdrantClient(host='localhost', port=6333)
cols = qc.get_collections()
for col in cols.collections:
    info = qc.get_collection(col.name)
    print(f'\nQdrant [{col.name}]: {info.points_count} points')
