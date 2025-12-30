from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import aiohttp
import chromadb
from chromadb.utils import embedding_functions
from typing import List

app = FastAPI()

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
request_queue = asyncio.Queue()

# --- MEMORY SETUP (ChromaDB) ---
# We connect to the same folder you just created
client = chromadb.PersistentClient(path="./vector_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="my_resume", embedding_function=embedding_func)

# --- HELPER: RETRIEVE CONTEXT ---
def get_relevant_context(query: str):
    # Ask the DB: "Find 2 chunks most similar to this query"
    results = collection.query(query_texts=[query], n_results=2)
    # Flatten the list of strings
    if results["documents"]:
        return "\n".join(results["documents"][0])
    return ""

# --- AI WORKER ---
async def call_ollama(session, user_query):
    # 1. RETRIEVE (Look up memory)
    context = get_relevant_context(user_query)
    
    # 2. AUGMENT (Create the "Smart" Prompt)
    # We tell the AI to behave like a Recruiter Assistant
    system_prompt = f"""
    You are a helpful assistant answering questions about a candidate named Abhyaung.
    Use the following Resume context to answer the question.
    If the answer is not in the context, say "I don't know."
    
    CONTEXT:
    {context}
    """
    
    full_prompt = f"System: {system_prompt}\nUser: {user_query}"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    # 3. GENERATE (Send to Brain)
    try:
        print(f"🧠 Thinking about: {user_query}...")
        async with session.post(OLLAMA_URL, json=payload) as response:
            result = await response.json()
            return result.get("response", "Error: No response")
    except Exception as e:
        return f"Error: {str(e)}"

async def run_llm_inference(batch: List[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [call_ollama(session, prompt) for prompt in batch]
        results = await asyncio.gather(*tasks)
    return results

# --- BACKGROUND WORKER (Unchanged) ---
async def batch_processor():
    print("🤖 RAG Batch Processor Started...")
    while True:
        batch = []
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
            print(f"✅ Processed {len(results)} items.")
            # Print the results to verify RAG is working
            for i, res in enumerate(results):
                print(f"Q: {batch[i]}")
                print(f"A: {res[:100]}...") # Print preview
        
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processor())

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    #Direct call
    async with aiohttp.ClientSession() as session:
        result = await call_ollama(session, request.prompt)
    return{"response":result}
