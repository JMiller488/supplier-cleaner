# Supplier Name Cleaner

A Python NLP project in two parts: a method evaluation comparing three name-matching techniques, and a practical cleaning tool built on the strongest of the three.

## The Problem

Procurement and finance datasets frequently contain the same supplier recorded under dozens of slightly different names — typos, abbreviations, legal suffixes, geographical name variations. This makes spend analysis and supplier consolidation unreliable without significant manual cleaning.

---

## Part 1 — Method Evaluation

Three approaches to supplier name matching were evaluated against a handcrafted dataset of 250 realistic supplier name variants (based on domain knowledge), covering abbreviations, acronyms, typos, and legal suffix variations. I wanted to build the complexity as I went, starting with Levenshtein and ending with a full embedding.

### The three methods

**Levenshtein distance**
The simplest baseline. Measures the minimum number of character edits (insertions, deletions, substitutions) required to transform one string into another, normalised to [0, 1] using `rapidfuzz`. Fast and fully interpretable, but purely character-level — it has no concept of meaning and struggles with names that differ structurally rather than by a few characters.

**TF-IDF with character n-grams**
The next level of complexity. Each supplier name is represented as a sparse vector of character n-gram (2–4 character) TF-IDF weights. Cosine similarity between these vectors then measures character-level overlap. No learned representations, fully interpretable, and near-instant to compute. No "knowledge" of the world outside of the dataset provided.

**Sentence Transformer embeddings**
The most complex model. Each name is mapped to some dense vector in R^384 using `all-MiniLM-L6-v2`. Semantically similar words will map to vectors pointing in roughly the same direction. We then compute the cosine similarity matrix on all pairs. Pairs scoring above a configurable threshold are connected as edges in a graph; connected components of that graph form the final groups, with the canonical name selected alphabetically for stability. Note, this embedding model has been pre-trained and finetuned on a large external corpus - therefore it is the first of our methods to incorporate some knowledge from outside the dataset provided.

### Results

All three methods were evaluated using a precision/recall sweep across thresholds from 0.0 to 1.0. Key findings:

- **Levenshtein** has the lowest overall performance — character edit distance is insufficient for the variety of real-world name variants
- **TF-IDF** achieves high precision but near-zero recall above τ = 0.75; it only works when names share significant character overlap
- **Sentence Transformer** maintains higher recall across a much wider threshold range, meaning it finds more true matches without sacrificing precision

My a priori estimate for a reasonable threshold was 0.85. See below for results.

Results at τ = 0.85 (initial estimate)
| Method | Precision | Recall |
|---|---|---|
| Sentence Transformer | 1.00 | 0.25 |
| TF-IDF | 1.00 | ~0.00 |
| Levenshtein | ~0.80 | ~0.05 |

This clearly is not a good threshold - the best model still misses 75% of real duplicates. Also, each model will perform at its best at different thresholds - so it is unfair to compare them all with an arbitrary threshold (0.85 in this case). 

We therefore want to pick the optimal threshold for each model and compare performance. Computing the F1 score over a range of thresholds and picking the threshold where F1 is maximal provides the optimal threshold for each model for a fair comparison.

Running this evaluation yields the following results at each model's F1-optimal threshold:

| Method | Optimal τ | Precision | Recall | F1 |
|---|---|---|---|---|
| Sentence Transformer | 0.69 | 0.91 | 0.63 | 0.74 |
| TF-IDF | 0.57 | 0.74 | 0.61 | 0.67 |
| Levenshtein | 0.77 | 0.68 | 0.30 | 0.41 |

**Conclusion:** The simpler baselines do not match the sentence transformer's performance, justifying the added complexity of a learned embedding model. The remaining recall gap is not a model choice issue — it reflects cases (acronyms, entity resolution) that require a fundamentally different approach.

The precision/recall curve can be regenerated with:
```bash
python analysis/scripts/run_evaluation.py
```

### Shared limitation

All three methods fail to resolve acronym mappings (e.g. PwC → PricewaterhouseCoopers, TCS → Tata Consultancy Services). There is no lexical or semantic signal connecting the abbreviated and full forms. A lookup table or dedicated entity resolution step would be required to resolve this gap.

### Threshold selection

The appropriate threshold depends on the cost of errors in context:

- **Higher threshold (e.g. τ = 0.85):** conservative — minimises false merges at the cost of missing genuine duplicates. Appropriate where incorrect grouping is more damaging than a missed one.
- **Lower threshold (e.g. τ = 0.65):** more aggressive — catches more duplicates but risks merging distinct suppliers.

To find the optimal threshold for a specific dataset, compute F1 at each threshold and select the maximum. Note that this score is dataset-specific — the optimal value depends on the noise characteristics of a particular supplier list.

---

## Part 2 — Cleaning Tool

Built on the sentence transformer approach validated in Part 1. Takes any CSV or Excel file as input and produces a cleaned version with supplier names grouped into canonical forms.

### Pipeline

The pipeline runs in three stages:

1. **Preprocessing** — rule-based normalisation: lowercasing, stop word removal (legal suffixes, generic business terms), Australian state name abbreviation, ordinal standardisation, punctuation stripping
2. **Embedding** — each preprocessed name is mapped to a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Grouping** — cosine similarity is computed across all vectors; pairs above a configurable threshold are connected as edges in a graph, connected components of that graph form the final groups, and the canonical name for each group is selected alphabetically for stability

### Why connected components?

A naive greedy approach assigns group labels based on whichever name appears first in the data — making results order-dependent and unstable. The connected components approach treats clustering as a graph problem: names are nodes, similarity above the threshold creates edges, and NetworkX identifies all clusters. The result is identical regardless of input order.

### Model choice

`all-MiniLM-L6-v2` was chosen because:

- **Speed** — at 22M parameters it encodes thousands of names in seconds, making it practical for interactive use
- **Short-text performance** — fine-tuned on semantic similarity tasks; performs well on the 2–5 word phrases that supplier names typically are
- **Accessibility** — small enough to run on a laptop without a GPU

### Web app

```bash
streamlit run app/app.py
```

Upload a CSV or Excel file. A local Mistral model (via Ollama) automatically detects the supplier name column — this can be overridden manually. Set the similarity threshold using the slider, then click Clean Suppliers. The output adds three columns to the original file: raw name, preprocessed name, and canonical group name. The file is downloadable as Excel.

Requires Ollama running locally with Mistral installed (supporting European tech!):

```bash
ollama pull mistral
```

---

## Project Structure

```
supplier-cleaner/
├── src/supplier_cleaner/
│   ├── preprocessing.py        # Text normalisation
│   ├── grouping.py             # Embedding + graph-based grouping
│   └── evaluate.py             # Precision/recall evaluation
├── analysis/
│   ├── scripts/
│   │   ├── run_evaluation.py   # Threshold sweep + method comparison plot
│   ├── data/
│   │   └── supplier_cleaning_synthetic_250_exact.csv  # Handcrafted dataset for model evaluation
│   └── results/
│       └── precision_recall_curve.png  # Latest precision/recall curve
├── app/
│   └── app.py                  # Streamlit web application
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_evaluate.py        # Evaluation unit tests
│   └── test_integration.py     # End-to-end pipeline test
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

# Run the web app
streamlit run app/app.py

# Regenerate the method comparison plot
python analysis/scripts/run_evaluation.py
```

**Requirements:** Python 3.10+, Ollama running locally with Mistral installed.

## Tech Stack

- `sentence-transformers` — text embeddings
- `networkx` — graph-based connected components grouping
- `streamlit` — web interface
- `pandas` — data handling
- `scikit-learn` — TF-IDF vectorisation (evaluation baseline)
- `rapidfuzz` — Levenshtein distance (evaluation baseline)
- `ollama` — local LLM for automatic column detection
- `matplotlib` — precision/recall curve plotting


