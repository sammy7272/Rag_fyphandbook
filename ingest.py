import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. Configuration ---
PDF_PATH = "3. FYP-Handbook-2023.pdf"  # Actual PDF filename
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings (250-400 words as per assignment)
# Smaller chunks for better precision
# Avg word is ~5 chars, so 200 words * 5 = 1000 chars
CHUNK_SIZE = 1000  # Reduced from 1500
# 20-40% overlap. Using 30%: 1000 * 0.30 = 300
CHUNK_OVERLAP = 300  # Reduced from 450

def main():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    print(f"Loading PDF: {PDF_PATH}...")
    # Load the PDF. This loader creates one doc per page,
    # and automatically stores the page number in metadata.
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print(f"Chunking documents (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    # Add required metadata: section_hint and chunk_id
    print("Adding metadata (section_hint, chunk_id)...")
    
    for idx, chunk in enumerate(chunks):
        # Add chunk_id
        chunk.metadata['chunk_id'] = f"chunk_{idx}"
        
        # Extract section_hint (first heading found in chunk)
        text = chunk.page_content
        # Look for headings - try multiple patterns
        section_hint = "Unknown"
        
        # Pattern 1: Lines that are all caps and short (likely headings)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line_stripped = line.strip()
            if len(line_stripped) > 0:
                # Check if it looks like a heading (short, might be numbered, might be all caps)
                if (len(line_stripped) < 80 and 
                    (line_stripped.isupper() or 
                     re.match(r'^\d+\.?\s+[A-Z]', line_stripped) or
                     re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$', line_stripped))):
                    section_hint = line_stripped[:100]  # Limit length
                    break
        
        # If no heading found, try to get first meaningful line
        if section_hint == "Unknown":
            for line in lines[:5]:
                line_stripped = line.strip()
                if len(line_stripped) > 10 and len(line_stripped) < 100:
                    section_hint = line_stripped[:100]
                    break
        
        chunk.metadata['section_hint'] = section_hint
    
    print("Metadata added to all chunks.")

    print(f"Initializing embedding model: {MODEL_NAME}...")
    # 3. Embed & Store
    # Use HuggingFaceEmbeddings to use the specified 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}  # Use CPU for simplicity
    )

    print("Creating FAISS index...")
    # Create the FAISS index from the chunks and their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"Saving index to {INDEX_PATH}...")
    # Save the index locally
    vector_store.save_local(INDEX_PATH)
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()