import os
from typing import List
from PyPDF2 import PdfReader
from docx import Document


def read_pdf(file_path: str) -> str:
    """Read and extract text from PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")
    return text.strip()


def read_docx(file_path: str) -> str:
    """Read and extract text from DOCX file."""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {str(e)}")
    return text.strip()


def read_txt(file_path: str) -> str:
    """Read and extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")


def process_uploaded_files(uploaded_files) -> str:
    """
    Process multiple uploaded files and return combined text.
    Supports PDF, DOCX, and TXT formats.
    """
    combined_text = ""
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = read_pdf(temp_path)
        elif file_ext == 'docx':
            text = read_docx(temp_path)
        elif file_ext == 'txt':
            text = read_txt(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: PDF, DOCX, TXT")
        
        combined_text += f"\n\n--- Content from {uploaded_file.name} ---\n\n{text}"
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return combined_text.strip()
