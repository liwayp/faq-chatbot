# Smart FAQ Chatbot (RAG)

AI-powered chatbot that reads corporate FAQ documents and answers user questions using RAG (Retrieval-Augmented Generation) architecture.

## Features

- 📁 Document Upload — PDF, DOCX, TXT support
- 🔍 Semantic Search — FAISS finds relevant fragments by meaning
- 🤖 GPT-4 Generation — Smart answers based on found context
- 💬 Chat Interface — User-friendly dialog with history
- 📚 Sources — Shows fragments used for answers
- 🗑️ History Clear — Reset chat button
- ⚠️ Safety — Honest responses when answer not in documents
- 💾 Persistent Storage — Index saved to disk, survives restarts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key-here
```

Or export it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Run the app:
```bash
streamlit run app.py
```

## Usage

1. **Upload FAQ Documents**: Use the sidebar to upload one or more FAQ files (PDF, DOCX, or TXT)
2. **Create Index**: Click the "Create FAISS Index" button to process documents
3. **Ask Questions**: Use the chat interface to ask questions about your FAQ content
4. **View Sources**: Click "View Sources" under any answer to see the original context
5. **Clear History**: Use the "Clear Chat History" button in the sidebar to reset the conversation

## Testing

A sample FAQ file (`sample_faq.txt`) is included for testing purposes. Simply upload this file and create the index to test the chatbot.

## Architecture

```
User Question
     │
     ▼
[Embedding Model] ──→ Question Vector
     │
     ▼
[FAISS Vector DB] ──→ Top-4 Relevant Chunks
     │
     ▼
[GPT-4] ──→ Final Answer Generation
     │
     ▼
Response to User
```

## Project Structure

```
BI/
├── app.py                      # Main Streamlit application
├── document_processor.py       # PDF, DOCX, TXT file parsing
├── vector_db.py               # FAISS vector database & embeddings
├── rag_pipeline.py            # RAG pipeline (retrieve + generate)
├── requirements.txt           # Python dependencies
├── sample_faq.txt            # Sample FAQ file for testing
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```
