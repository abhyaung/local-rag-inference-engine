# Local RAG Inference Engine 🧠

A high-performance, privacy-focused AI Chat System running entirely on local hardware (Apple Silicon). This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Llama-3**, served via an asynchronous **FastAPI** backend with dynamic request batching.

## 🚀 Architecture Overview

This is not just a wrapper around an API. It is a full-stack inference engine designed for high concurrency and low latency.

### The Stack
* **Hardware:** Optimized for Apple M-Series Chips (Metal/GPU acceleration).
* **Model Engine:** [Ollama](https://ollama.com/) running **Meta Llama 3 (8B)**.
* **Backend:** **FastAPI** (Python) handling async request ingestion.
* **Orchestration:** Custom **Producer-Consumer Queue** for request batching.
* **Memory (RAG):** **ChromaDB** for vector storage and semantic search.
* **Frontend:** **Streamlit** for a ChatGPT-like web interface.

## 📸 Screenshots

<p align="center">
  <img src="snapshots/image1.png" alt="LandingPage" width="30%"/> 
  <img src="snapshots/image2.png" alt="LandingPage2" width="30%"/>
  <img src="snapshots/image3.png" alt="DashBoard" width="30%"/>
  <img src="snapshots/image4.png" alt="BookingPage" width="30%"/>
</p>

### System Flow
1.  **Ingestion:** User queries are received via the Streamlit UI and sent to the API.
2.  **Buffering:** Requests are validated and pushed into an `asyncio.Queue` (Non-blocking).
3.  **Retrieval:** The system searches the local Vector Database (ChromaDB) for relevant context (e.g., private PDFs).
4.  **Batch Processing:** A background worker pulls requests and processes them in batches to maximize GPU throughput.
5.  **Inference:** The augmented prompt is sent to Llama-3 via local HTTP request.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed
* An Apple Silicon Mac (M1/M2/M3/M4) is recommended for GPU acceleration.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/local-rag-inference-engine.git](https://github.com/YOUR_USERNAME/local-rag-inference-engine.git)
cd local-rag-inference-engine
