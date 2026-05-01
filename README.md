# 🤖 Agentic AI + RAG using LangGraph

A portfolio project that demonstrates how to build an **Agentic AI system with Retrieval-Augmented Generation (RAG)** using LangGraph, Google Gemini, FAISS, and Streamlit.

This project allows an AI agent to retrieve information from documents (TXT/PDF), use tools dynamically, and generate context-aware responses.

---

## 🚀 Features

- Agentic workflow using **LangGraph**
- Retrieval-Augmented Generation (**RAG**)
- Document support for **TXT and PDF**
- Vector search with **FAISS**
- Google Gemini LLM integration
- Custom embedding fix for Gemini
- Tool calling with LangChain tools
- Streamlit chat interface
- LangSmith tracing and debugging

---

## 🛠 Tech Stack

- **Python**
- **LangGraph**
- **LangChain**
- **Google Gemini**
- **FAISS**
- **Streamlit**
- **LangSmith**

---

## 📂 Project Structure

- `main.py` → Core LangGraph + RAG logic  
- `app.py` → Streamlit user interface  
- `data.txt` → Text knowledge base  
- `data.pdf` → PDF knowledge base  
- `.env` → API keys and configuration  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/rag-langgraph.git
cd rag-langgraph
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

---

## ▶️ Run Project

Run CLI version:

```bash
uv run main.py
```

Run Streamlit UI:

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Load documents from TXT or PDF  
2. Split text into chunks  
3. Generate embeddings  
4. Store vectors in FAISS  
5. Retrieve relevant context  
6. AI agent decides whether to use tools  
7. Generate final response using retrieved context  

---

## 📊 Observability

This project uses **LangSmith tracing** to monitor:

- Agent steps
- Tool calls
- LLM responses
- Debugging workflow

---

## 💡 Use Cases

- AI document chatbots  
- Knowledge assistants  
- Research assistants  
- Customer support bots  
- Private document Q&A systems  

---

## 🔮 Future Improvements

- Chat memory  
- Streaming responses  
- Multi-tool agents  
- Database integration  
- Multi-document ingestion  

---

## 👨‍💻 Author

**Talha Ishaq**  
AI Engineer | LangGraph Developer | Agentic AI Builder  

---

## 📜 License

MIT License
