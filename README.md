# 🎯 Local AI Interview Assistant

A 100% free, privacy-focused, local AI-powered interview practice platform built using **Streamlit**, **Ollama**, **ChromaDB (RAG)**, **Faster-Whisper (STT)**, and **Web Speech API / gTTS / pyttsx3 (TTS)**.

---

## 🌟 Features

- 🔄 **Dual Interview Modes**:
  - **Practice Specific Round**: Practice single rounds tailored to HR, Technical, Coding, or System Design.
  - **Full Mock Interview**: Automated multi-stage interview experience progressing across HR -> Technical -> Coding -> System Design.
- 📚 **Resume & Job Description RAG Context**:
  - Extracts text from uploaded PDF/TXT files or direct text input.
  - Generates vector embeddings using local Ollama (`nomic-embed-text`) stored in **ChromaDB** for context-aware questions and evaluations.
- 🎤 **Voice & Audio Capabilities**:
  - **Speech-to-Text (STT)**: Record your answers using your microphone and transcribe them locally using `faster-whisper`.
  - **Text-to-Speech (TTS)**: Built-in voice assistant with **`🔊 Read Question Aloud`** and **`🛑 Stop`** buttons for interactive audio readout using Web Speech API and `gTTS`/`pyttsx3`.
- 💻 **Real Python Code Execution**:
  - Automated sandbox runner executes candidate code against test cases for the **Coding Agent**.
- 📊 **Hybrid Scoring & Comprehensive Reports**:
  - Evaluates candidate answers with a combination of LLM scoring, keyword matching, semantic similarity, and code execution results.
  - Tracks performance history, weak areas, and auto-generates structured learning plans and downloadable PDF reports.

---

## 🛠️ System Architecture & Tech Stack

- **Frontend / UI**: Streamlit
- **Local LLM & Embeddings**: Ollama (`gemma2:2b`, `qwen2.5-coder:3b`, `nomic-embed-text`)
- **Vector Database**: ChromaDB
- **Speech-to-Text**: Faster-Whisper
- **Text-to-Speech**: Browser Web Speech API, gTTS, pyttsx3
- **Database**: SQLite
- **Report Generation**: ReportLab (PDF)

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have installed:
- **Python 3.10+**
- **Ollama**: [Download & Install Ollama](https://ollama.com/)

### 2. Download Required Ollama Models

Run the following commands in your terminal to pull the local AI models:

```bash
# Language models for question generation and evaluation
ollama pull gemma2:2b
ollama pull qwen2.5-coder:3b

# Embedding model for RAG indexing
ollama pull nomic-embed-text
```

### 3. Installation

Clone or navigate to the project directory and install python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Launch the Streamlit app:

```bash
streamlit run app/ui/main.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## 📁 Project Structure

```
├── app/
│   ├── agents/               # Interview agents (HR, Technical, Coding, System Design, Evaluation, Report)
│   ├── analytics/            # Weak area tracking & performance analytics
│   ├── database/             # SQLite storage for interview logs
│   ├── execution/            # Local Python code execution runner
│   ├── llm/                  # Ollama local API integration client
│   ├── nlp/                  # Scoring algorithms & hybrid evaluation
│   ├── rag/                  # ChromaDB vector indexer & document splitter
│   ├── speech/               # Transcriber (Whisper) & Text-to-Speech (TTS)
│   ├── ui/                   # Streamlit user interface (main.py)
│   └── utils/                # JSON utilities, fallback generators, file extractors
├── docs/                     # Project documentation
├── main.txt                  # Consolidated project blueprint
├── requirements.txt          # Python dependencies
└── README.md                 # Project README documentation
```

---

## 🔒 Privacy & Local First

All processing—including text embeddings, LLM inference, voice transcription, code execution, and database storage—runs 100% locally on your machine. No external API keys or cloud subscriptions required!