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

## Project Structure
```
supplier-cleaner/
├── src/
│   └── supplier_cleaner/
│       ├── preprocessing.py   # Text normalisation
│       ├── grouping.py        # Embedding + graph-based grouping
│       └── app.py             # Streamlit web application
├── notebooks/
│   └── exploration.ipynb      # Original exploratory analysis
├── tests/
│   └── test_preprocessing.py  # Unit tests
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