import sys
sys.path.insert(0, 'src')
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import FieldCondition, Filter, MatchValue
from config.settings import EMBEDDING_MODEL

qc = QdrantClient(host='localhost', port=6333)
em = SentenceTransformer(EMBEDDING_MODEL)
query = "How has the performance of DHFL been for the financial year 2017-18?"
qv = em.encode("Represent this sentence: " + query, convert_to_tensor=False).tolist()

res = qc.query_points(
    collection_name='financial_documents',
    query=qv,
    query_filter=Filter(must=[FieldCondition(key='company_id', match=MatchValue(value=1))]),
    limit=8,
)

total_chars = sum(len(p.payload.get('chunk_text', '') or '') for p in res.points)
print(f"Chunks returned: {len(res.points)}")
print(f"Total chunk text chars: {total_chars}")
print(f"Estimated prompt tokens (rough): {total_chars // 4}")
for p in res.points:
    ct = p.payload.get('chunk_text', '') or ''
    doc_type = p.payload.get('document_type', '')
    print(f"  score={p.score:.3f}  chars={len(ct):5d}  type={doc_type}")
