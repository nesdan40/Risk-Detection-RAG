from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@127.0.0.1:5433/financial_risk_db"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database();"))
    print(result.scalar())