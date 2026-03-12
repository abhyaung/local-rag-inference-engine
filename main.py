import subprocess
from fastapi import UploadFile, File, FastAPI
from pydantic import BaseModel
import asyncio
import aiohttp
import chromadb
from chromadb.utils import embedding_functions
from typing import List
from contextlib import asynccontextmanager
from pypdf import PdfReader
import io

# --- CONFIGURATION ---
# Port 11434 is the default for Ollama
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Verify this matches your 'ollama list' output (e.g., "llama3.2" or "llama3.1")
MODEL_NAME = "llama3.2" 
request_queue = asyncio.Queue()

# --- LIFESPAN MANAGER (Modern replacement for @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background worker
    print("🤖 RAG Batch Processor Starting...")
    asyncio.create_task(batch_processor())
    yield
    # Shutdown logic can go here if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# --- MEMORY SETUP (ChromaDB) ---
client = chromadb.PersistentClient(path="./vector_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# We use get_or_create here to avoid the "Collection not found" crash
collection = client.get_or_create_collection(name="my_resume", embedding_function=embedding_func)

# --- HELPER: RETRIEVE CONTEXT ---
def get_relevant_context(query: str):
    results = collection.query(query_texts=[query], n_results=3) # Increased results to 3
    if results["documents"] and len(results["documents"][0]) > 0:
        context = "\n".join(results["documents"][0])
        print(f"📄 Found context ({len(context)} chars)") # DEBUG LINE
        return context
    print("⚠️ No relevant resume context found!") # DEBUG LINE
    return ""

# --- AI WORKER ---
async def call_ollama(session, user_query):
    context = get_relevant_context(user_query)
    
    system_prompt = f"""
    You are a helpful assistant answering questions about a candidate named Abhyaung.
    Use the following Resume context to answer the question.
    If the answer is not in the context, say "I don't know."
    
    CONTEXT:
    {context}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"System: {system_prompt}\nUser: {user_query}",
        "stream": False
    }

    try:
        print(f"🧠 Querying {MODEL_NAME} for: {user_query[:50]}...")
        async with session.post(OLLAMA_URL, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("response", "Error: No response key found")
            else:
                return f"Error: Ollama returned status {response.status}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

async def run_llm_inference(batch: List[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [call_ollama(session, prompt) for prompt in batch]
        results = await asyncio.gather(*tasks)
    return results

# --- BATCH PROCESSOR ---
async def batch_processor():
    while True:
        batch = []
        # Try to pull up to 4 items from the queue
        while len(batch) < 4:
            try:
                if batch:
                    item = request_queue.get_nowait()
                else:
                    item = await request_queue.get()
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        
        if batch:
            results = await run_llm_inference(batch)
            print(f"✅ Batch of {len(results)} processed.")
        
        await asyncio.sleep(0.1)

# --- ENDPOINTS ---
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    # For RAG, we call our worker directly to get the response back to the user
    async with aiohttp.ClientSession() as session:
        result = await call_ollama(session, request.prompt)
    return {"response": result}

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    print(f"📥 Receiving file: {file.filename}")
    
    # 1. Save the uploaded file to your project folder, replacing the old 'resume.pdf'
    file_location = "resume.pdf"
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
        
    print("💾 File saved to disk. Triggering ingest.py...")

    # 2. Command the system to run your existing ingest.py script
    try:
        # Note: If your system requires './.venv/bin/python3', swap it in the line below
        result = subprocess.run(
            ["./.venv/bin/python3", "ingest.py"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Ingestion Output:\n{result.stdout}")
        
        # 3. Force ChromaDB client in main.py to reload the newly updated database
        global collection
        collection = client.get_collection(name="my_resume")
        
        return {"message": "Document uploaded and processed successfully via ingest.py!"}
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ingestion Failed:\n{e.stderr}")
        return {"error": "Failed to run ingest.py. Check backend logs."}
