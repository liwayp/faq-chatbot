import os
from typing import List, Tuple
from openai import OpenAI


class RAGPipeline:
    """RAG Pipeline: Retrieve + Generate using FAISS and GPT-4."""
    
    def __init__(self, vector_db, model: str = "gpt-4"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_db = vector_db
        self.model = model
        self.system_prompt = """You are a helpful FAQ chatbot that answers questions based on the provided context. 
Your responses should be:
- Accurate and based ONLY on the provided context
- Clear and concise
- Professional and friendly

If the answer cannot be found in the context, respond with: 
"I don't have enough information in my knowledge base to answer that question. Please contact support for assistance."

Do not make up answers or provide information outside of the given context."""
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks from vector database."""
        return self.vector_db.search(query, top_k)
    
    def generate_answer(self, query: str, context_chunks: List[Tuple[str, float]]) -> Tuple[str, List[str]]:
        """Generate answer using GPT-4 based on retrieved context."""
        # Format context
        context_text = "\n\n".join([
            f"Context {i+1}:\n{chunk}" 
            for i, (chunk, _) in enumerate(context_chunks)
        ])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the user's question.

{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract source chunks for display
            sources = [chunk for chunk, _ in context_chunks]
            
            return answer, sources
            
        except Exception as e:
            return f"Error generating response: {str(e)}", []
    
    def query(self, user_question: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """Full RAG pipeline: retrieve + generate."""
        # Retrieve
        context_chunks = self.retrieve(user_question, top_k)
        
        if not context_chunks:
            return "I couldn't find any relevant information. Please try rephrasing your question or contact support.", []
        
        # Generate
        answer, sources = self.generate_answer(user_question, context_chunks)
        
        return answer, sources
