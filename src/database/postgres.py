from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import POSTGRES


def get_engine():
    db_url = (
        f"postgresql+psycopg2://"
        f"{POSTGRES['user']}:{POSTGRES['password']}@"
        f"{POSTGRES['host']}:{POSTGRES['port']}/"
        f"{POSTGRES['database']}"
    )

    # Connection string — password masked in logs
    masked_url = db_url.replace(POSTGRES["password"], "***")
    # Uncomment to debug: print(masked_url)

    engine = create_engine(
        db_url,
        echo=False,
        future=True
    )

    return engine


def test_connection():
    """
    Tests the PostgreSQL connection.
    """

    try:
        engine = get_engine()
        
        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT current_database();")
            )

            database_name = result.scalar()

            print(f"✓ PostgreSQL connected to '{database_name}'.")

        print("✓ PostgreSQL connection successful.")

    except SQLAlchemyError as e:
        print("✗ PostgreSQL connection failed.")
        print(e)