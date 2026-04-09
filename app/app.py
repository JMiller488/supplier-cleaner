"""
app.py
------
Streamlit web application for the supplier name cleaning pipeline.
Provides a UI for uploading data, selecting the supplier column,
running the cleaning pipeline, and downloading results.
"""

import io

import pandas as pd
import streamlit as st

from supplier_cleaner.preprocessing import preprocess_supplier_name
from supplier_cleaner.grouping import group_suppliers


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Supplier Cleaner", page_icon="🏭", layout="wide")


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

    unique_names = df["Supplier preprocessed"].unique().tolist()
    supplier_groups = group_suppliers(unique_names, threshold=threshold)

    df["Supplier grouped"] = df["Supplier preprocessed"].map(supplier_groups)
    return df


# ── UI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the Streamlit application."""
    st.title("Supplier Name Cleaner")
    st.caption(
        "Upload an Excel or CSV file, pick the supplier column, "
        "and download a cleaned and grouped output."
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

    # ── Column selection ──────────────────────────────────────────────────────
    st.subheader("Select supplier column")
    supplier_col = st.selectbox(
        "Which column contains the supplier names?",
        options=list(df.columns),
    )

    st.write("**5 sample values from selected column:**")
    st.write(df[supplier_col].dropna().astype(str).head(5).tolist())

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("Settings")
    threshold = st.slider(
        "Similarity threshold for grouping",
        min_value=0.50,
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