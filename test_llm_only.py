import sys, time
sys.path.insert(0, 'src')
from ui.ollama_client import generate
from ui.rag_query import _build_prompt

# Simulate 5 chunks of 800 chars each
fake_chunks = [
    {
        "chunk_text": "DHFL reported total revenue of Rs 8,500 crores in FY2018, a growth of 22% over the previous year. Profit after tax stood at Rs 1,020 crores. The company maintained a healthy NIM of 2.8%." * 4,
        "document_type": "annual_report",
        "fiscal_year": "FY2018",
        "fiscal_quarter": None,
        "file_path": "data/raw/DHFL/annual_reports/DHFL_AnnualReport_FY2018.pdf",
    }
] * 5

prompt = _build_prompt(
    "How has the performance of DHFL been for the financial year 2017-18?",
    fake_chunks,
)
print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

start = time.time()
answer = generate(prompt)
elapsed = time.time() - start
print(f"LLM response in {elapsed:.1f}s")
print(f"Answer: {answer[:300]}")
