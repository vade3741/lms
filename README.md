# 🎯 Local AI Interview Assistant

A 100% free, privacy-focused, local AI-powered interview practice platform built using **Streamlit**, **Ollama**, **ChromaDB (RAG)**, **Faster-Whisper (STT)**, and **Web Speech API / gTTS / pyttsx3 (TTS)**.

---

## 🌟 Features

- 🔄 **Dual Interview Modes**:
  - **Practice Specific Round**: Practice single rounds tailored to HR, Technical, Coding, or System Design.
  - **Full Mock Interview**: Automated multi-stage interview experience progressing across HR -> Technical -> Coding -> System Design.
- 📚 **Resume & Job Description RAG Context**:
  - Extracts text from uploaded PDF/TXT files or direct text input.
  - Generates vector embeddings using local Ollama (`embeddinggemma:latest` by default) stored in **ChromaDB** for context-aware questions and evaluations.
- 📖 **PDF Study & Revision**:
  - Upload multiple PDFs or sync the local `pdf/` folder; uploaded files are limited to 50 MB each.
  - Search passages, generate short source-grounded notes, and create interview questions with document/page citations.
  - Practise saved questions one at a time, receive evidence-based feedback, and see due reviews scheduled from answer scores.
  - Documents, notes, questions, answers, and revision history stay in local SQLite/Chroma storage; removing a PDF also removes its dependent questions and practice history.
- ✍️ **Personal question bank**: Add your own interview questions and reference answers for offline practice.
- 🌐 **Optional public-source research**:
  - Search public web results with the free `ddgs` package, import readable article excerpts with `trafilatura`, and generate locally saved questions from them.
  - Internet access is only needed for search/import; previously saved material remains in the local app.
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
- **Local LLM & Embeddings**: Ollama (`qwen2.5-coder:7b` and `embeddinggemma:latest` by default; configurable)
- **Vector Database**: ChromaDB
- **Speech-to-Text**: Faster-Whisper
- **Text-to-Speech**: Browser Web Speech API, gTTS, pyttsx3
- **Database**: SQLite
- **Optional free web research**: `ddgs` and `trafilatura` (no paid search/API key)
- **Report Generation**: ReportLab (PDF)

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have installed:
- **Python 3.10+**
- **Ollama**: [Download & Install Ollama](https://ollama.com/)

### 2. Ollama models

The app uses the locally installed `qwen2.5-coder:7b` for generation and
`embeddinggemma:latest` for embeddings by default. You can reuse those models if
they are already installed. Check availability with:

```bash
ollama list
```

If a configured model is missing, install it with `ollama pull <model-name>`.
The app sidebar checks the Ollama service and configured models and shows the
exact pull commands for any missing models. Keep the Ollama app running while
using model-backed features.

### 3. Installation

Clone or navigate to the project directory and install python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional local configuration can be set before launching the app:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="qwen2.5-coder:7b"
export OLLAMA_EMBEDDING_MODEL="embeddinggemma:latest"
```

Other supported settings are `OLLAMA_TIMEOUT_SECONDS`,
`OLLAMA_EMBEDDING_TIMEOUT_SECONDS`, `AI_INTERVIEW_DB_PATH`,
`AI_INTERVIEW_CHROMA_PATH`, and `AI_INTERVIEW_PDF_DIRECTORY`. Data paths default
to locations under this project. Configuration is read from environment
variables; do not put private content or secrets into shared shell startup files.

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

To use **Study & Revision**, upload text-based PDFs in the app or place them in the project's `pdf/` folder and choose **Sync PDF folder**. Scanned image-only PDFs currently need OCR before they can be searched. PDF notes, question generation, and answer feedback require the configured local Ollama models. Public-source search/import is optional and requires internet access; local PDFs and previously imported sources/questions remain stored on your computer. Some sites block article downloads; on HTTP 403, the app can save the attributed search-result snippet instead and clearly marks it as a snippet with less context than the article.

### A practical daily revision loop

1. Add your PDFs once from **PDF library & notes**.
2. Pick a topic and create short cited notes and a small batch of grounded questions.
3. Open **Practice & due review**, answer before revealing the sample answer, then review the source citations and missing concepts.
4. Use **Progress & history** to revisit answers and follow the score-based due queue.
5. Add your own questions or optionally search public material, import an article excerpt, and create questions from that saved source.

The due-review queue uses simple score-based intervals: low scores return after one day; stronger scores progress through 1, 3, 7, 14, and (for scores of 90 or higher) 30 days. This is a transparent revision heuristic, not a clinical or scientifically validated learning assessment.

Run the offline tests from the project directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

---

## 📁 Project Structure

```
├── app/
│   ├── agents/               # Interview agents (HR, Technical, Coding, System Design, Evaluation, Report)
│   ├── analytics/            # Weak area tracking & performance analytics
│   ├── database/             # SQLite storage for interview and study history
│   ├── execution/            # Local Python code execution runner
│   ├── llm/                  # Ollama local API integration client
│   ├── nlp/                  # Scoring algorithms & hybrid evaluation
│   ├── rag/                  # ChromaDB vector indexer & document splitter
│   ├── study_workflow.py     # PDF study, public sources, practice, revision
│   ├── speech/               # Transcriber (Whisper) & Text-to-Speech (TTS)
│   ├── ui/                   # Streamlit user interface (main.py)
│   └── utils/                # JSON utilities, fallback generators, file extractors
├── pdf/                      # Local PDFs used by Study & Revision
├── docs/                     # Project documentation
├── main.txt                  # Consolidated project blueprint
├── requirements.txt          # Python dependencies
└── README.md                 # Project README documentation
```

---

## 🔒 Privacy & Local First

PDF text extraction, embeddings, question/note generation, practice feedback,
voice transcription, and study-history storage run locally using the configured
Ollama models and local files/databases. Optional public-source search and
article import make network requests when explicitly used. Existing text-to-
speech helpers may also use online services depending on their fallback path;
do not send private PDF text to those optional features.