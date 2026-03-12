import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

# 1. ChromaDB Setup
client = chromadb.PersistentClient(path="./vector_db")

# Ensure this matches exactly with what you use in main.py
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ingest_resume(file_path):
    print(f"📄 Reading {file_path}...")

    # 2. Reader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            # Replace newlines with spaces so we don't break sentences
            text += extracted.replace('\n', ' ') + " " 
    
    if len(text.strip()) < 10:
        print("❌ Error: Extracted text is too short. Is the PDF a scanned image or empty?")
        return

    print(f"    - Extracted {len(text)} characters")

    # 3. Improved Chunking (Sliding Window by words)
    words = text.split()
    chunk_size = 150  # Words per chunk
    overlap = 30      # Overlap to keep company names attached to bullet points
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50: # Only ignore truly empty garbage chunks
            chunks.append(chunk)

    print(f"    - Split into {len(chunks)} overlapping chunks.")

    # 4. Wipe the old database memory and recreate it
    try:
        client.delete_collection("my_resume")
        print("    - Cleared old database memory.")
    except Exception:
        pass # Ignore if it doesn't exist yet
        
    collection = client.create_collection(
        name="my_resume",
        embedding_function=sentence_transformer_ef
    )

    # 5. Store in VectorDB
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"source": "resume"} for _ in chunks]

    print("💾 Saving to Vector Database...")
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Ingestion complete! {collection.count()} items now in the database.")

if __name__ == "__main__":
    # Check for both cases to be safe
    target_file = "resume.pdf"
    if not os.path.exists(target_file):
        target_file = "Resume.pdf"

    if os.path.exists(target_file):
        ingest_resume(target_file)
    else:
        print(f"❌ Error: No resume file found in directory. Please ensure '{target_file}' exists.")
