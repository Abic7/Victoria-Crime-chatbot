# 🔍 Local RAG Assistant — Offline CSV Q&A with Gradio + Local LLMs

> Ask natural questions about your CSV data using semantic search and a local large language model. 100% offline. No OpenAI keys. No cloud dependencies.

![Gradio Screenshot](<img width="1088" height="981" alt="image" src="https://github.com/user-attachments/assets/25d7ab06-8905-492c-bd0e-46651141c46e" />) <!-- Replace with actual screenshot if available -->

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
git clone https://github.com/yourusername/local-rag-assistant.git
cd local-rag-assistant
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Requirements</summary>

```text
pandas
numpy
faiss-cpu
sentence-transformers
requests
gradio
```

</details>

---

### 3. Start your LLM locally

You can use any LLM served via OpenAI-compatible API. Recommended:

* 🔥 [LM Studio](https://lmstudio.ai/) – supports many Hugging Face models

> Example local endpoint: `http://localhost:1234/v1/chat/completions`

---

### 4. Launch the Gradio app

```bash
python gradio_rag_app.py
```

Then open your browser to `http://localhost:7860`

---

## 🧪 Example Questions (Crime Dataset)

```text
"What is the most stolen item in Abbotsford in 2025?"

"How many food-related items were stolen in postcode 3067?"

"List total value of items stolen in commercial zones in Q1 2025."
```

---

## 🧠 Prompt Engineering (Under the Hood)

Every user question generates a rich, structured prompt like:

```text
You are a data analysis assistant. The following is a list of rows from a crime dataset...

ONLY use the data provided. Do NOT make up any data.

### Question:
What is the most stolen item in Abbotsford in 2025?

### Data:
<Row 1>
<Row 2>
...

### Instructions:
- Use rows that match the filters in the question.
- If nothing matches, say "No data found."
```

This ensures **grounded, explainable outputs** — ideal for high-trust domains like auditing, law enforcement, or finance.

---

## 📌 Use Cases

* 🕵️ Crime and safety data analysis
* 📈 Financial transaction review
* 🧾 Policy compliance checks
* 🧠 No-code internal BI assistants
* 📁 Log + report analysis (GDPR, HIPAA, ISO, etc.)

---

## 🔐 Why Offline?

* ✅ Privacy by default
* 💸 Zero token/API costs
* 🔄 Total control over data and model
* 🔍 Fully explainable outputs with traceable context

---

## 🌍 Future Improvements

* [ ] 🔍 Add support for PDFs, DOCX
* [ ] 📊 Integrate interactive charts with Plotly
* [ ] 🧠 Enable memory-aware follow-up Q\&A
* [ ] 💾 Export answers to PDF/CSV
* [ ] 🔄 Replace Gradio with Streamlit option

---

## 🤝 Contributing

Got ideas? Want to adapt this for your industry? PRs are welcome!

```bash
# Fork + star ⭐
# Create a branch 🚀
# Submit a PR with improvements 💡
```

---

## 🙋‍♂️ Author & License

Built by [Abhc7](https://www.linkedin.com/in/abichaudhuri/)
MIT License © 2025

---

## 🌟 Acknowledgments

* HuggingFace Transformers + SentenceTransformers
* Facebook FAISS
* Gradio Team
* LM Studio / Ollama contributors
* All open-source AI pioneers 🙌

---

## 🔗 Stay Connected

💬 [LinkedIn](https://www.linkedin.com/in/abichaudhuri/)
📩 Open to collaborations, demos, and talks!

---

> 🧠 Sometimes, the best AI isn’t in the cloud — it’s sitting quietly on your machine, waiting to be asked the right question.

```

