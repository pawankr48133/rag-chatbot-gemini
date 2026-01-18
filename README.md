# 🤖 RAG Chatbot powered by 🔗 LangChain, Gemini & Hugging Face 🤗

<div align="center">
  <img src="https://raw.githubusercontent.com/your-username/rag-chatbot-gemini/main/screenshots/rag_architecture.png">
  <figcaption>RAG architecture implemented using LangChain, Gemini, and ChromaDB.</figcaption>
</div>

---

## Project Overview <a name="overview"></a>

Although Large Language Models (LLMs) such as Gemini are powerful and capable of generating high-quality responses, they **cannot natively access private or user-specific documents** and may hallucinate or provide incomplete answers.

To overcome this limitation, **Retrieval-Augmented Generation (RAG)** is used.  
RAG enhances LLMs by retrieving **relevant external documents** and injecting them into the model’s context before generating an answer.

The aim of this project is to build a **RAG-based chatbot using LangChain** powered by:

- **Gemini-2.5 Flash** for answer generation  
- **Hugging Face local embeddings** for semantic search  
- **ChromaDB** as a vector database  

Users can upload documents in **PDF, TXT, CSV, or DOCX** format and interactively chat with their data.  
Relevant document chunks are retrieved using vector similarity search and passed to Gemini for **grounded, context-aware responses**.

The project follows **modern LangChain best practices** using **LangChain Expression Language (LCEL)** and provides an interactive UI built with **Streamlit**.

---

## RAG Architecture <a name="architecture"></a>

