from qdrant_client import QdrantClient

from config.settings import QDRANT


def get_client():
    """
    Creates and returns a Qdrant client.
    """

    client = QdrantClient(
        host=QDRANT["host"],
        port=QDRANT["port"]
    )

    return client


def test_connection():
    """
    Tests connectivity with the Qdrant server.
    """

    try:
        client = get_client()

        collections = client.get_collections()

        print("✓ Qdrant connection successful.")
        print(f"Collections found: {len(collections.collections)}")

    except Exception as e:
        print("✗ Qdrant connection failed.")
        print(e)