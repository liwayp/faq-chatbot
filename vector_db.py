import os
import uuid
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from typing import List, Tuple
from openai import OpenAI


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using OpenAI API."""
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, input: List[str]) -> Embeddings:
        response = self.client.embeddings.create(
            model=self.model,
            input=input
        )
        return [data.embedding for data in response.data]


class VectorDatabase:
    """ChromaDB vector database for storing and retrieving document chunks."""

    PERSIST_DIR = "./chroma_db"

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=embedding_model
        )
        self.chroma_client = chromadb.PersistentClient(path=self.PERSIST_DIR)
        self.collection = None
        self.chunks = []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI API."""
        return self.embedding_fn(texts)

    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1].strip()
                    end = start + break_point + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - overlap

        return chunks

    def create_index(self, text: str, progress_callback=None) -> Tuple[int, List[str]]:
        """Create ChromaDB collection from text and return chunk count and chunks."""
        self.chunks = self.split_text_into_chunks(text)

        if progress_callback:
            progress_callback(30, "Splitting documents...")

        try:
            self.chroma_client.delete_collection(name="faq_collection")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="faq_collection",
            embedding_function=self.embedding_fn
        )

        batch_size = 100
        all_ids = []

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            all_ids.extend([str(uuid.uuid4()) for _ in batch])

            if progress_callback:
                progress = 30 + int((i / len(self.chunks)) * 60)
                progress_callback(progress, f"Creating embeddings... ({i}/{len(self.chunks)})")

        if progress_callback:
            progress_callback(90, "Adding to ChromaDB...")

        self.collection.add(
            documents=self.chunks,
            ids=all_ids
        )

        if progress_callback:
            progress_callback(100, "Index created successfully!")

        return len(self.chunks), self.chunks

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Search for most relevant chunks given a query."""
        if self.collection is None or len(self.chunks) == 0:
            raise ValueError("No index created yet. Please create an index first.")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        result_list = []
        if results['documents'] and results['distances']:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0.0
                result_list.append((doc, distance))

        return result_list

    def save_index(self):
        """No explicit save needed - PersistentClient auto-saves to disk."""
        pass

    def load_index(self):
        """Load ChromaDB collection from disk."""
        try:
            self.collection = self.chroma_client.get_collection(
                name="faq_collection",
                embedding_function=self.embedding_fn
            )
            all_docs = self.collection.get()
            self.chunks = all_docs['documents'] if all_docs else []
            return True
        except:
            return False

    def is_ready(self) -> bool:
        """Check if vector database is ready."""
        return self.collection is not None and len(self.chunks) > 0
