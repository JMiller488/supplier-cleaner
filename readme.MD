# Supplier Name Cleaner

A Python NLP pipeline for deduplicating and grouping messy supplier names using semantic similarity.

## Problem

Procurement and finance datasets frequently contain the same customer or supplier recorded under dozens of slightly different names — typos, abbreviations, legal suffixes, geographical name variations. This makes spend analysis and supplier consolidation unreliable without significant manual cleaning.

## Approach

The pipeline runs in three stages:

1. **Preprocessing** — rule-based normalisation: lowercasing, stop word removal, Australian state name abbreviation, ordinal standardisation, and punctuation stripping
2. **Embedding** — each preprocessed name is encoded into a vector using a sentence transformer model (`all-MiniLM-L6-v2` as default)
3. **Grouping** — a cosine similarity matrix is computed across all name vectors; pairs above a configurable threshold are connected as edges in a graph; connected components of that graph form the final groups, with the canonical name selected alphabetically for stability

## Why connected components?

A naive greedy approach assigns group labels based on whichever name appears first in the data — making results order-dependent and unstable. The connected components approach treats the problem as a graph problem: names are nodes, similarity above the threshold creates edges, and NetworkX finds all clusters. The result is identical regardless of input order.

## Model choice

The pipeline uses `all-MiniLM-L6-v2` as the default sentence transformer. This model was chosen because:

- **Speed** — at 22M parameters it encodes thousands of names in seconds, making it practical for interactive use in the Streamlit app
- **Short-text performance** — it was fine-tuned on semantic similarity tasks and performs well on short phrases, which is exactly what supplier names are
- **Size** — the model is small enough to run on a laptop without a GPU, keeping the project accessible

Larger models (e.g. `all-mpnet-base-v2`) may capture more nuance but are significantly slower — a poor tradeoff for this use case where names are typically 2-5 words.

## Threshold selection

The default similarity threshold of 0.85 was selected by evaluating precision and recall on a synthetic ground-truth dataset of labelled supplier name pairs. A threshold sweep across 101 values (0.0–1.0) showed that 0.85 provides a strong balance: high precision (few false merges) with good recall (most true duplicates caught). The precision-recall curve is saved to `data/precision_recall_curve.png` and can be regenerated with `python scripts/run_evaluation.py`.

## Project Structure
```
supplier-cleaner/
├── src/supplier_cleaner/
│   ├── preprocessing.py        # Rule-based text normalisation
│   ├── grouping.py             # Embedding + graph-based grouping
│   ├── evaluate.py             # Precision/recall evaluation framework
│   └── app.py                  # Streamlit web application
├── scripts/
│   ├── generate_synthetic.py   # Synthetic ground truth generation
│   └── run_evaluation.py       # Threshold sweep + plotting
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_evaluate.py        # Evaluation unit tests
│   └── test_integration.py     # End-to-end pipeline test
├── data/
│   ├── synthetic_pairs.csv     # Labelled name pairs
│   └── precision_recall_curve.png
├── notebooks/                  # Exploratory analysis
├── requirements.txt
└── README.md
```

## Quickstart
```bash
git clone https://github.com/JMiller488/supplier-cleaner.git
cd supplier-cleaner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/supplier_cleaner/app.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with Mistral (supporting European tech!) installed (`ollama pull mistral`)

## Tech Stack

- `sentence-transformers` — text embeddings
- `networkx` — graph-based connected components grouping
- `streamlit` — web interface
- `pandas` — data handling
- `ollama` — local LLM for automatic column detection