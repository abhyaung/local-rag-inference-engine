import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

#1. ChromaDB Setup
client = chromadb.PersistentClient(path="./vector_db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
            name = "my_resume",
            embedding_function = sentence_transformer_ef
        )

def ingest_resume(file_path):
    print(f"📄Reading {file_path}...")

    #2. Reader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text +=  page.extract_text()
    
    print(f"    -Extracted {len(text)} characters")

    #3. Chunking
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"    -Split into {len(chunks)} chunks.")

    #4. Store in VectorDB
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"source": "resume"} for _ in chunks]

    print("💾Saving to Vector Database...")
    collection.add(
            documents = chunks,
            ids = ids,
            metadatas = metadatas
    )
    print("✅ Ingetion complete! Your AI now remembers this file.")

if __name__ == "__main__":
    if os.path.exists("Resume.pdf"):
        ingest_resume("Resume.pdf")
    else:
        print("❌Error: 'resume.pdf' not found in the directory")
    

