from qdrant_client import QdrantClient

client: QdrantClient | None = None



def get_qdrant_client():
    global client
    if client is None:
        init_client()
    return client

def init_client() -> QdrantClient:
    global client
    if client is None:
        print("Initializing QDrant Client...")
        client = QdrantClient(path="./qdrant_storage")