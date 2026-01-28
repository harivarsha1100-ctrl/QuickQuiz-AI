# 🚀 QuickQuiz-AI — Smart AI MCQ Generator

**Version:** 1.0.0  

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal.svg)]()
[![React](https://img.shields.io/badge/React-Frontend-blue.svg)]()

--- 

## 📌 Project Overview

**QuickQuiz-AI** is an advanced **AI-powered Multiple Choice Question (MCQ) generation system** that automatically converts educational content into high-quality quizzes.

The system leverages **Transformer-based deep learning models** to generate:

- Intelligent questions  
- Accurate answers  
- Meaningful distractors  
- Difficulty classification  
- Bloom’s Taxonomy mapping  

QuickQuiz-AI significantly reduces manual effort in assessment creation and provides a scalable solution for **online education, examinations, and learning analytics**.

---

## 🎯 Key Features

✅ Upload educational content (PDF, DOCX, TXT)  
✅ Automatic question generation using **Transformer models**  
✅ Context-aware answer extraction  
✅ Intelligent distractor generation  
✅ Difficulty-level classification (Easy / Medium / Hard)  
✅ Bloom’s Taxonomy tagging  
✅ Explanation generation for answers  
✅ Quiz evaluation and scoring  
✅ Export results to **PDF and DOCX**  
✅ Modular and production-ready backend  
✅ Modern React frontend interface  

---

## 📑 Table of Contents

1. [Quick Start](#quick-start)
2. [Models](#models)
3. [Architecture](#architecture)
4. [File Structure](#file-structure)
5. [Model Cards & Paths](#model-cards--paths)
6. [Backend API Reference](#backend-api-reference)
7. [Important Modules](#important-modules--how-they-interact)
8. [Distractor Model Integration](#distractor-model-integration-details)
9. [Appendices](#appendices)

---

## 🚀 Quick Start

> Assumes you have cloned the repository and are inside the `backend/` directory.

### 1️⃣ Create Python virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
# if SentencePiece needed
pip install sentencepiece
```

### 3) Start backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

Access the UI at `http://localhost:5173` (frontend) and backend at `http://127.0.0.1:8000`.

---

## 🧠 Models

⚠️ Trained transformer models are **not included** in this repository due to size constraints.

You can:
- Download pretrained models from Hugging Face
- Fine-tune models using your own datasets
- Place models locally under `backend/MCQ_MODEL_Check/`

Model paths are configurable in the backend.

```
MCQ_MODEL_Check/
  qg_merged/                # merged QG model (t5-small merged weights)
    config.json
    model.safetensors
    spiece.model (or spm.model)
    tokenizer_config.json
    tokenizer.json
  QA_Model/qa_Model/         # QA model (distilbert-based saved via save_pretrained)
    config.json
    pytorch_model.bin / model.safetensors
    tokenizer files
  distractors_model_merged/  # merged distractor generator (optional)
    config.json
    model.safetensors
    spiece.model
    tokenizer_config.json
```

> **Important:** `local_files_only=True` is used when loading models from local paths. Ensure the directory names match exactly (case-sensitive on Linux/mac).

## Architecture — diagram + explanation

```
flowchart LR
  A[Frontend: React (Vite, Tailwind)] -->|REST| B[FastAPI Backend]
  B --> C[QuestionGenerator (QG model: T5-small + LoRA)]
  B --> D[Answer Extraction (QA model: distilbert-base-uncased)]
  B --> E[DistractorGenerator (T5-based or heuristics)]
  B --> F[Supabase (Postgres)]
  B --> G[Exporter (PDF/DOCX)]
  style C fill:#f3f4f6,stroke:#333,stroke-width:1px
  style D fill:#eef2ff,stroke:#333,stroke-width:1px
  style E fill:#fff7ed,stroke:#333,stroke-width:1px
```

### Explanation

- **Frontend (React)**: Uploads files, requests question generation and quiz sessions, displays quizzes and exports results.
- **FastAPI backend**: Orchestrator — receives uploads, extracts text (PyPDF2, python-docx), calls QuestionGenerator, stores questions in Supabase, and serves API endpoints for quiz lifecycle.
- **QuestionGenerator**: Uses the fine-tuned QG T5-small model (LoRA merged weights or merged model) to produce question text. If QA model is available, uses QA pipeline to extract answers; otherwise falls back on heuristics.
- **DistractorGenerator**: Separate module to generate 3 distractors using either a dedicated merged T5 distractor model or heuristic rules. Kept separate for modularity.
- **Supabase**: Stores uploaded file metadata, extracted text, generated questions, sessions, responses, and exports.
- **Exporter**: Produces PDF/DOCX using ReportLab and python-docx.

---

## File structure

```
project/
├── backend/
│   ├── main.py                     # FastAPI entrypoint
│   ├── requirements.txt
│   ├── .env
│   ├── modules/
│   │   ├── file_processor.py       # PDF/DOCX parsing
│   │   ├── question_generator.py   # QG orchestration (uses QG model)
│   │   ├── distractor_generator.py # Distractor model wrapper
│   │   ├── quiz_evaluator.py       # scoring + storing responses
│   │   ├── exporter.py             # export to pdf/docx
│   │   └── database.py             # Supabase client
│   └── MCQ_MODEL_Check/             # local models (recommended path)
│       ├── qg_merged/               # QG merged model (t5-small merged) — used by QuestionGenerator
│       ├── QA_Model/qa_Model/       # QA model (distilbert-base-uncased fine-tuned) — used by pipeline
│       └── distractors_model_merged/ # distractor merged model
├── src/                       # React frontend
│   ├── App.tsx               # Main application
│   ├── components/           # React components
│   │   ├── FileUpload.tsx
│   │   ├── QuestionGenerator.tsx
│   │   ├── QuizInterface.tsx
│   │   └── ResultsDisplay.tsx
│   ├── services/
│   │   └── api.ts           # API client
│   └── config/ 
│       └── supabase.ts      # Supabase config
│
├── .env                      # Environment variables
├── package.json             # Node dependencies
└── README.md               # This file                      
```

> Keep `MCQ_MODEL_Check/` inside backend so `local_files_only=True` loads from local disk without hitting HF hub.

---

## Model cards & paths

This project uses three model types.

### 1) Question Generation (QG)
- **Model**: `t5-small + LoRA (merged)`
- **Recommended directory**: `backend/MCQ_MODEL_Check/qg_merged/`
- **Files expected**: `config.json`, `model.safetensors` or `pytorch_model.bin`, `spiece.model` or `spm.model`, `tokenizer_config.json`, `tokenizer.json` (or `vocab` files)
- **Load code** (in `question_generator.py`):

```py
from transformers import T5ForConditionalGeneration, T5Tokenizer
self.t5_model_name = "MCQ_MODEL_Check/qg_merged"
self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name, local_files_only=True)
self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name, local_files_only=True)
```

### 2) Question Answering (QA)
- **Model**: `distilbert-base-uncased` (fine-tuned for QA)
- **Recommended directory**: `backend/MCQ_MODEL_Check/QA_Model/qa_Model/`
- **Files expected**: typical `save_pretrained()` output for model + tokenizer
- **Load code** (pipeline):

```py
from transformers import pipeline
self.qa_pipeline = pipeline(
    "question-answering",
    model="MCQ_MODEL_Check/QA_Model/qa_Model",
    tokenizer="MCQ_MODEL_Check/QA_Model/qa_Model",
    device=self.device if self.device >= 0 else -1
)
```

> If you get errors like `Repo id must be in the form 'namespace/repo_name'`, double-check your path and use `local_files_only=True` or provide an absolute path.

### 3) Distractor Generator (DG)
- **Model**: T5-small fine-tuned for generating distractors (merged weights)
- **Recommended directory**: `backend/MCQ_MODEL_Check/distractors_model_merged/`
- **Files expected**: `config.json`, `model.safetensors`, `spiece.model`, `tokenizer_config.json`.
- **Loading approach**: Create `distractor_generator.py` that exposes a `DistractorGenerator` class which loads this merged model and has a method `generate(question, context, num=3)`.

---

## Backend API reference (endpoints)

> Example requests shown are for the FastAPI backend (`main.py`).

### `GET /` — Health check
- Response: `{ status: "online", service: "Smart AI MCQ Generator" }`

### `POST /api/upload` — Upload file
- Form-data: `file` (PDF/DOCX/TXT), optional `user_id`
- Response: `file_id`, `preview`, `text_length`

### `POST /api/generate-questions` — Generate questions
- Form-data: `file_id`, `num_questions`, `difficulty` (optional)
- Orchestrates: extract text → QuestionGenerator.generate_mcqs → store questions
- Response: list of generated questions (with options and correct answer)

### `POST /api/create-quiz-session` — Create a quiz session
- Form-data: `file_id`, `user_id`, `session_name`
- Response: `session_id` and questions without correct answers

### `POST /api/submit-quiz` — Submit quiz responses
- Form-data: `session_id`, `answers` (JSON string)
- Response: evaluation report, percentage, per-question results

### `POST /api/export` — Export results
- Form-data: `session_id`, `export_type` (`questions_only` or `results_with_answers`), `file_format` (`pdf` or `docx`)
- Response: Downloadable file

---

## Important modules & how they interact

### `modules/question_generator.py`
- Loads QG model (local merged t5-small) and QA pipeline (local distilbert QA model) if available.
- Public method: `generate_mcqs(text, num_questions, difficulty)` which returns list of question dicts:

```py
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "correct": "B"},
  "correct_answer": "B",
  "explanation": "...",
  "difficulty": "Easy",
  "blooms_taxonomy": "Remember",
  "topic": "..."
}
```

- Uses `modules/distractor_generator.py` to get 3 distractors. If that module is missing or the model cannot be loaded, the generator falls back to heuristic distractors.

### `modules/distractor_generator.py`
- Encapsulates the distractor model. Example interface:

```py
class DistractorGenerator:
    def __init__(self, model_path: str = "MCQ_MODEL_Check/distractors_model_merged"):
        # load tokenizer + model locally
    def generate(self, question_text: str, context: str, num: int = 3) -> List[str]:
        # returns list of distractor strings
```

**Where to import:** In `question_generator.py` use `from modules.distractor_generator import DistractorGenerator` (ensure `modules` is a package with `__init__.py`).

---

## Distractor model integration details

### Recommended pattern (in `distractor_generator.py`)

```py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class DistractorGenerator:
    def __init__(self, model_dir="MCQ_MODEL_Check/distractors_model_merged"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = T5Tokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)

    def generate(self, question: str, context: str, num: int = 3):
        prompt = f"generate distractors: question: {question} context: {context}"
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        outs = self.model.generate(inputs["input_ids"], max_length=64, num_return_sequences=1, do_sample=True)
        decoded = self.tok.decode(outs[0], skip_special_tokens=True)
        # parse returned distractors (expects 'd1 | d2 | d3')
        parts = [p.strip() for p in decoded.split("|") if p.strip()]
        return parts[:num]
```

### Where to place the code

- Save `distractor_generator.py` into `backend/modules/` next to `question_generator.py`.
- Make sure `backend/modules/__init__.py` exists, so you can import as `from modules.distractor_generator import DistractorGenerator`.

## Appendices

### Checklist before running locally

- [ ] `backend/MCQ_MODEL_Check/qg_merged/` exists and contains `config.json` + tokenizer files + `model.safetensors` or `pytorch_model.bin`.
- [ ] `backend/MCQ_MODEL_Check/QA_Model/qa_Model/` exists and contains QA model & tokenizer saved via `save_pretrained()`.
- [ ] `backend/MCQ_MODEL_Check/distractors_model_merged/` exists for distractor generation (optional — heuristics fallback exists).
- [ ] `backend/modules/__init__.py` exists so `modules` is a package.
- [ ] Virtual environment activated and `requirements.txt` installed.

---

## 👨‍💻 Author

Developed with ❤️ as an AI/ML Final Year Project
Project Name: QuickQuiz-AI — Smart MCQ Generator
---

