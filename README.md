# AI-Chatbox

An AI-powered chatbot built using Streamlit, Sentence Transformers, FAISS, and GPT-2. The application combines semantic search and Retrieval-Augmented Generation (RAG) to provide context-aware responses from a custom knowledge base.

---

## Overview

AI-chatbox is an intelligent chatbot that retrieves relevant information from a knowledge base using vector search and generates responses using a pre-trained language model.

The system uses semantic embeddings, FAISS indexing, and GPT-2 text generation to improve response quality and contextual understanding.

---

## Features

### AI Chat Interface

* Interactive chatbot interface built with Streamlit
* Real-time user interaction
* Chat history management

### Semantic Search

* Sentence Transformers embeddings
* Semantic similarity matching
* Context retrieval from knowledge base

### Retrieval-Augmented Generation (RAG)

* Retrieves relevant information before generating responses
* Context-aware answer generation
* Improved response relevance

### Language Model Integration

* GPT-2 text generation model
* Dynamic response generation
* Natural language interaction

### Conversational Features

* Session-based chat memory
* Clear chat functionality
* Loading indicator during response generation

### Knowledge Base Support

* Custom text-based knowledge source
* Automatic document embedding
* Fast document retrieval

---

## Technologies Used

* Python
* Streamlit
* Sentence Transformers
* FAISS
* NumPy
* Hugging Face Transformers
* GPT-2

---

## Project Workflow

1. Load Knowledge Base
2. Generate Document Embeddings
3. Create FAISS Vector Index
4. Retrieve Relevant Context
5. Generate AI Response using GPT-2
6. Display Response in Chat Interface

---

## Project Structure

```text
app.py
knowledge_base.txt
```

---

## How to Run

### Install Dependencies

```bash
pip install streamlit sentence-transformers faiss-cpu numpy transformers torch
```

### Run Application

```bash
streamlit run app.py
```

---

## Key Highlights

* AI-Chatbot
* Semantic Search Engine
* Retrieval-Augmented Generation (RAG)
* Vector Database Search using FAISS
* Context-Aware Responses
* GPT-2 Based Text Generation
* Interactive Streamlit Interface

---

## Author

Mohit Bansal
