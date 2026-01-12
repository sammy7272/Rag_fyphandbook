# FYP Handbook RAG Assistant

This project is a RAG (Retrieval-Augmented Generation) application designed to answer questions from the FYP Handbook.

## structure

- `app.py`: Main Streamlit application.
- `ingest.py`: Script to ingest the PDF and create the vector embedding index.
- `requirements.txt`: Python dependencies.
- `3. FYP-Handbook-2023.pdf`: Source document.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the embeddings:
   ```bash
   python ingest.py
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```
