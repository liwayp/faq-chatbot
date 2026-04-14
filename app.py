import streamlit as st
import os
from document_processor import process_uploaded_files
from vector_db import VectorDatabase
from rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)


def update_progress(progress, message):
    """Update progress bar."""
    st.session_state.progress = progress
    st.session_state.progress_message = message
    st.progress(progress, text=message)


def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'index_ready' not in st.session_state:
        st.session_state.index_ready = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'progress_message' not in st.session_state:
        st.session_state.progress_message = ""
    
    # Auto-load saved index if exists
    if st.session_state.vector_db is None and not st.session_state.index_ready:
        try:
            vector_db = VectorDatabase()
            if vector_db.load_index():
                st.session_state.vector_db = vector_db
                st.session_state.rag_pipeline = RAGPipeline(vector_db)
                st.session_state.index_ready = True
        except:
            pass  # No saved index found


def sidebar_admin():
    """Admin sidebar for document upload and management."""
    with st.sidebar:
        st.header("📋 Admin Panel")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload FAQ Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Create index button
        if uploaded_files:
            if st.button("🔨 Create FAISS Index", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        # Process uploaded files
                        progress_container = st.empty()
                        
                        def progress_callback(progress, message):
                            progress_container.progress(progress / 100)
                            progress_container.text(message)
                        
                        combined_text = process_uploaded_files(uploaded_files)
                        
                        # Create vector database
                        vector_db = VectorDatabase()
                        chunk_count, chunks = vector_db.create_index(
                            combined_text,
                            progress_callback=progress_callback
                        )
                        
                        st.session_state.vector_db = vector_db
                        
                        # Save index to disk (auto-saved by PersistentClient)
                        vector_db.save_index()
                        
                        # Create RAG pipeline
                        st.session_state.rag_pipeline = RAGPipeline(vector_db)
                        st.session_state.index_ready = True
                        
                        progress_container.empty()
                        st.success(f"✅ Index created successfully! ({chunk_count} chunks)")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating index: {str(e)}")
        
        # Index status
        st.divider()
        if st.session_state.index_ready:
            st.success("✅ Knowledge Base Ready")
        else:
            st.warning("⚠️ No knowledge base loaded")
        
        # Recent messages
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📝 Recent Messages")
            recent = st.session_state.chat_history[-5:]
            for msg in recent:
                with st.expander(f"{'👤' if msg['role'] == 'user' else '🤖'} {msg['role']}"):
                    st.write(msg['content'][:100] + "...")
        
        # Clear chat button
        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Delete saved index
        if st.session_state.index_ready:
            st.divider()
            if st.button("🗑️ Delete Saved Index", type="secondary"):
                try:
                    import shutil
                    if os.path.exists(VectorDatabase.PERSIST_DIR):
                        shutil.rmtree(VectorDatabase.PERSIST_DIR)
                    st.session_state.vector_db = None
                    st.session_state.rag_pipeline = None
                    st.session_state.index_ready = False
                    st.session_state.chat_history = []
                    st.success("✅ Index deleted from disk!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting index: {str(e)}")


def display_chat_message(msg):
    """Display chat message with formatting."""
    with st.chat_message(msg['role']):
        st.write(msg['content'])
        
        # Show sources if assistant message with sources
        if msg['role'] == 'assistant' and msg.get('sources'):
            with st.expander("📚 View Sources"):
                for i, source in enumerate(msg['sources'], 1):
                    st.markdown(f"**Source {i}:**")
                    st.info(source[:300] + "..." if len(source) > 300 else source)
                    st.divider()


def main():
    """Main application."""
    initialize_session_state()
    
    # Sidebar
    sidebar_admin()
    
    # Main area
    st.title("🤖 Smart FAQ Chatbot")
    st.caption("Ask me anything about our FAQ documents!")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        display_chat_message(msg)
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Check if index is ready
        if not st.session_state.index_ready:
            st.error("⚠️ Please upload FAQ documents and create the index first!")
            return
        
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_history.append(user_msg)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.rag_pipeline.query(prompt)
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.info(source[:300] + "..." if len(source) > 300 else source)
                                st.divider()
                    
                    # Add assistant message to history
                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                    st.session_state.chat_history.append(assistant_msg)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })


if __name__ == "__main__":
    main()
