import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from .embeddings import get_embeddings_model
import uuid

load_dotenv()

get_embedding = get_embeddings_model()


class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        existing_indexes = self.pc.list_indexes()
        if self.index_name not in [idx["name"] for idx in existing_indexes]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

    def upsert_document(self, text: str, metadata: dict):
        vector = get_embedding.embed_query(text)
        doc_id = str(uuid.uuid4())
        self.index.upsert(
            vectors=[
                {
                    "id": doc_id,
                    "values": vector,
                    "metadata": metadata,
                }
            ]
        )

    def query_documents(self, query_text: str, top_k=3):
        query_vector = get_embedding.embed_query(query_text)
        results = self.index.query(
            vector=query_vector, top_k=top_k, include_metadata=True
        )
        return [
            {"score": result.score, **result.metadata} for result in results.matches
        ]
