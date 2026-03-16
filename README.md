# genai1 — Prompt evaluation for synthetic data (imbalanced classes)

Streamlit app + notebooks for generating and evaluating LLM-created synthetic rows to help balance an imbalanced tabular dataset (e.g., credit-card default). The app uses the Groq Chat Completions API to generate new samples for the minority class and validates the generated CSV against the uploaded dataset’s schema.

## What’s included

- **Streamlit app**: upload CSV → pick target column → generate/edit a prompt template → generate synthetic data → download validated CSV.
- **Notebooks**: dataset preparation, train/test creation, and model prediction experiments.

## Project structure

- [app.py](app.py) — Streamlit UI + Groq call + CSV parsing/validation.
- [requirements.txt](requirements.txt) — Python dependencies.
- [data/](data) — datasets
  - [data/UCI_Credit_Card.csv](data/UCI_Credit_Card.csv) — source dataset
  - [data/train_data/train.csv](data/train_data/train.csv) — training split
  - [data/train_data/augmented_generated.csv](data/train_data/augmented_generated.csv) — example augmented data
  - [data/test_data/test.csv](data/test_data/test.csv) — test split
- [output/](output) — saved prompt/baseline outputs (text files)
- Notebooks
  - [building-dataset.ipynb](building-dataset.ipynb)
  - [create_train_test_data.ipynb](create_train_test_data.ipynb)
  - [model_perdiction.ipynb](model_perdiction.ipynb)

## Prerequisites

- Python 3.9+ (recommended)
- A Groq API key

## Setup

1) Create/activate a virtual environment (optional if you already have one):

```powershell
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate
```

2) Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3) Configure environment variables:

- Copy `.env.template` → `.env`
- Set `GROQ_API_KEY` in `.env` (the app loads it via `python-dotenv`).

## Run the application

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Notes / assumptions

- The target column is expected to be **binary (0/1)** for the app’s class-splitting logic.
- The app attempts to parse a CSV (preferably in a fenced ```csv block) and aligns columns/types to the uploaded dataset.

## Troubleshooting

- **`GROQ_API_KEY is not set`**: ensure `.env` exists at repo root and contains `GROQ_API_KEY=...`.
- **CSV parse / schema mismatch**: check the “Raw LLM output” section and ensure the generated CSV header exactly matches the uploaded file’s columns.
