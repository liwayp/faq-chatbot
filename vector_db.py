import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple
from openai import OpenAI


class VectorDatabase:
    """FAISS vector database for storing and retrieving document chunks."""

    PERSIST_DIR = "./chroma_db"

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.dimension = 1536

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [data.embedding for data in response.data]

    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

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
        """Create FAISS index from text and return chunk count and chunks."""
        self.chunks = self.split_text_into_chunks(text)

        if progress_callback:
            progress_callback(30, "Splitting documents...")

        batch_size = 100
        all_embeddings = []

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.extend(embeddings)

            if progress_callback:
                progress = 30 + int((i / len(self.chunks)) * 60)
                progress_callback(progress, f"Creating embeddings... ({i}/{len(self.chunks)})")

        if progress_callback:
            progress_callback(90, "Building FAISS index...")

        embeddings_array = np.array(all_embeddings).astype('float32')
        self.dimension = embeddings_array.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)

        if progress_callback:
            progress_callback(100, "Index created successfully!")

        return len(self.chunks), self.chunks

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Search for most relevant chunks given a query."""
        if self.index is None or len(self.chunks) == 0:
            raise ValueError("No index created yet. Please create an index first.")

        query_embedding = self.get_embeddings([query])
        query_vector = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(distances[0][i])))

        return results

    def save_index(self):
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        os.makedirs(self.PERSIST_DIR, exist_ok=True)
        faiss.write_index(self.index, self.PERSIST_DIR + "/faiss.index")
        with open(self.PERSIST_DIR + "/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        """Load FAISS index and chunks from disk."""
        try:
            self.index = faiss.read_index(self.PERSIST_DIR + "/faiss.index")
            with open(self.PERSIST_DIR + "/chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            return True
        except:
            return False

    def is_ready(self) -> bool:
        """Check if vector database is ready."""
        return self.index is not None and len(self.chunks) > 0
