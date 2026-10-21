import sys, time
sys.path.insert(0, 'src')
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL
from ui.rag_query import query_company

qc = QdrantClient(host='localhost', port=6333)
em = SentenceTransformer(EMBEDDING_MODEL)

start = time.time()
result = query_company(
    query_text="How has the performance of DHFL been for the financial year 2017-18?",
    company_id=1,
    qdrant_client=qc,
    embedding_model=em,
)
elapsed = time.time() - start

print(f"Completed in {elapsed:.1f}s")
print(f"\nANSWER:\n{result['answer']}")
print(f"\nSOURCES ({len(result['sources'])}):")
for i, s in enumerate(result['sources'], 1):
    import os
    base = os.path.basename(s.get('file_path') or '')
    print(f"  [{i}] {base} | {s.get('document_type')} | {s.get('fiscal_year')} {s.get('fiscal_quarter') or 'Annual'}")
    print(f"       {(s.get('chunk_text') or '')[:120]}...")
