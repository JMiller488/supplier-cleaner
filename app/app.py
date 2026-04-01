"""
app.py
------
Streamlit web application for the supplier name cleaning pipeline.
Provides a UI for uploading data, detecting the supplier column,
running the cleaning pipeline, and downloading results.
"""

import io

import ollama
import pandas as pd
import streamlit as st

from supplier_cleaner.preprocessing import preprocess_supplier_name
from supplier_cleaner.grouping import group_suppliers


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Supplier Cleaner", page_icon="🏭", layout="wide")


# ── Column detection ──────────────────────────────────────────────────────────


def detect_supplier_column(df: pd.DataFrame) -> str | None:
    """Use Mistral via Ollama to identify the supplier name column.

    Sends column names and sample values to a local Mistral model,
    which returns its best guess at which column contains supplier names.
    Falls back to None if detection fails or the response is ambiguous.

    Args:
        df: Input DataFrame.

    Returns:
        Column name string if detected, otherwise None.
    """
    samples = {col: df[col].dropna().astype(str).head(5).tolist() for col in df.columns}
    prompt = f"""You are analyzing a spreadsheet. Here are the column names and 5 sample values from each:

{samples}

Which column contains supplier or vendor company names?
Respond with ONLY the exact column name, nothing else."""

    try:
        response = ollama.chat(
            model="mistral", messages=[{"role": "user", "content": prompt}]
        )
        detected = response["message"]["content"].strip().strip('"').strip("'")
        return detected if detected in df.columns else None
    except ConnectionError:
        return "OLLAMA_UNAVAILABLE"
    except Exception:
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_pipeline(df: pd.DataFrame, supplier_col: str, threshold: float) -> pd.DataFrame:
    """Run the full supplier cleaning pipeline on a DataFrame.

    Adds three new columns to the input DataFrame:
        - 'Supplier': raw supplier name copied from the selected column
        - 'Supplier preprocessed': normalised name after preprocessing
        - 'Supplier grouped': canonical group name after similarity grouping

    Args:
        df: Input DataFrame.
        supplier_col: Name of the column containing raw supplier names.
        threshold: Cosine similarity threshold for grouping.

    Returns:
        DataFrame with three new columns appended.
    """
    df = df.copy()
    df["Supplier"] = df[supplier_col].astype(str)
    df["Supplier preprocessed"] = df["Supplier"].apply(preprocess_supplier_name)

    unique_names = df["Supplier preprocessed"].tolist()
    supplier_groups = group_suppliers(unique_names, threshold=threshold)

    df["Supplier grouped"] = df["Supplier preprocessed"].map(supplier_groups)
    return df


# ── UI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the Streamlit application."""
    st.title("Supplier Name Cleaner")
    st.caption(
        "Upload any Excel or CSV file — Mistral will detect the supplier column automatically."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your file here, or click to browse", type=["xlsx", "csv"]
    )

    if not uploaded_file:
        return

    # ── Load file ─────────────────────────────────────────────────────────────
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("File preview")
    st.dataframe(df.head(5), use_container_width=True)

    # ── Column detection ──────────────────────────────────────────────────────
    st.subheader("Column detection")
    with st.spinner("Asking Mistral to identify the supplier column..."):
        detected = detect_supplier_column(df)

    if detected == "OLLAMA_UNAVAILABLE":
        detected = None
        st.info(
            "Ollama is not running — column auto-detection is disabled. "
            "Select the supplier column manually below."
        )
    elif detected:
        st.success(f"Mistral identified **{detected}** as the supplier column.")
    else:
        st.warning("Mistral could not confidently detect the supplier column.")

    supplier_col = st.selectbox(
        "Confirm or change the supplier column:",
        options=list(df.columns),
        index=list(df.columns).index(detected) if detected else 0,
    )

    st.write("**5 sample values from selected column:**")
    st.write(df[supplier_col].dropna().astype(str).head(5).tolist())

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("Settings")
    threshold = st.slider(
        "Similarity threshold for grouping",
        min_value=0.70,
        max_value=0.99,
        value=0.69,
        step=0.01,
        help="Higher = only very similar names are grouped together",
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if not st.button("Clean Suppliers", type="primary"):
        return

    with st.spinner("Preprocessing supplier names..."):
        result_df = run_pipeline(df, supplier_col, threshold)

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original unique suppliers", result_df["Supplier"].nunique())
    col2.metric("After preprocessing", result_df["Supplier preprocessed"].nunique())
    col3.metric("After grouping", result_df["Supplier grouped"].nunique())

    st.subheader("Sample of grouped names")
    sample = (
        result_df[["Supplier", "Supplier preprocessed", "Supplier grouped"]]
        .drop_duplicates(subset="Supplier")
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(sample, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.subheader("Download cleaned file")
    output = io.BytesIO()
    result_df.to_excel(output, index=False)
    st.download_button(
        label="Download cleaned Excel",
        data=output.getvalue(),
        file_name="cleaned_suppliers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
