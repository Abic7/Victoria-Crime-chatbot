# 🔍 Local RAG Assistant — Offline CSV Q&A with Gradio + Local LLMs

> Ask natural questions about your CSV data using semantic search and a local large language model. 100% offline. No OpenAI keys. No cloud dependencies.

![Gradio Screenshot](https://your-screenshot-url.com) <!-- Replace with actual screenshot if available -->

---

## ✨ Overview

This project is a lightweight, offline-first **Retrieval-Augmented Generation (RAG)** system built with:

- 🧠 **Sentence Transformers** for semantic embeddings  
- ⚡ **FAISS** for high-speed vector search  
- 🤖 **Local LLMs** via [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com)  
- 🎛️ **Gradio** for an intuitive, browser-based UI

**Upload your CSV → Ask a question → Get a grounded, explainable answer.**

---

## 🔧 Key Features

✅ Upload any CSV (e.g., crime reports, transactions, logs)  
🔍 Ask natural language questions  
📄 Retrieve semantically relevant rows using FAISS  
🧪 Auto-generate a structured, context-aware prompt  
🤖 Get an answer from a **local LLM**  
🔐 100% offline — No internet or cloud APIs  
📊 Full transparency: shows answer, context used, and raw prompt

---

## 📦 Tech Stack

| Component        | Tool/Library                     |
|------------------|----------------------------------|
| UI               | [Gradio](https://www.gradio.app/)  
| Embeddings       | [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)  
| Vector Search    | [FAISS](https://github.com/facebookresearch/faiss)  
| LLM Backend      | [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com)  
| Programming Lang | Python 🐍  

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/Abic7/Victoria-Crime-chatbot.git
cd Victoria-Crime-chatbot
