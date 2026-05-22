# Chat with PDF - RAG System

A RAG (Retrieval Augmented Generation) system that lets you 
chat with any PDF document using LangChain and free LLMs.

## How it works
1. Load a PDF document
2. Split into chunks
3. Create embeddings using HuggingFace
4. Store in FAISS vector database
5. Ask questions and get answers using LLM via OpenRouter

## Libraries Used
- LangChain
- HuggingFace Embeddings
- FAISS
- OpenRouter (free LLM)

## How to run
1. Clone the repo
2. Install requirements: pip install -r requirements.txt
3. Add your OpenRouter API key in .env file
4. Run chat_with_pdf.ipynb