"""
Gene Process Dependency Explorer
Streamlit app for exploring DepMap gene dependency latent representations.
Reproduces the analysis pattern from:
  - WayScience/gene-process-dependencies 8.shiny-app/1.latent-pcas.ipynb  (PCA plots)
  - WayScience/gene-process-dependencies 5.drug-dependency/8.spider_plots.ipynb (spider plots)

Data schema mirrors the real DepMap / BioBombe outputs:
  - Models: PCA, ICA, NMF, DAE, VAE, β-VAE, β-TCVAE
  - Metadata: primary_disease, lineage, cell_line (CCLE name)
  - PCA columns: pca_1, pca_2 (per model + latent_dim)
  - Spider columns: mean dependency score per biological process category
"""

import colorsys
import pathlib
import random
import sys
import textwrap

import ipywidgets as widgets
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import streamlit as st
from app_utils import (
    clean_label,
    generate_random_palette,
    latent_load_data,
    load_data,
    load_model_data,
    make_dropdown_pca_with_selection,
    make_radar,
    place_labels_polar,
    single_load_data,
    spider_load_data,
    truncate_label,
)
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gene Process Dependency Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .stApp { background: #0d1117; color: #e6edf3; }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 28px; font-weight: 600; color: #58a6ff; font-family: 'IBM Plex Mono', monospace; }

  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
  h1 { color: #58a6ff !important; font-size: 1.6rem !important; }
  h2 { color: #79c0ff !important; font-size: 1.1rem !important; }
  h3 { color: #cdd9e5 !important; font-size: 0.95rem !important; }

  .section-tag {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #8b949e;
    margin-bottom: 12px;
  }

  div[data-testid="stSidebar"] {
    background: #010409 !important;
    border-right: 1px solid #21262d !important;
  }

  div[data-testid="stSidebar"] .stSelectbox label,
  div[data-testid="stSidebar"] .stMultiSelect label,
  div[data-testid="stSidebar"] .stSlider label {
    color: #8b949e !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 6px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Selections")
    st.markdown("---")
    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()

    # ── Disease selector ──────────────────────────────────────────────
    all_diseases = list(reactome_matrix["OncotreePrimaryDisease"].unique())

    st.markdown(
        "<span style=\"font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:0.5px;\">Primary diseases</span>",
        unsafe_allow_html=True,
    )
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("All", key="disease_all", use_container_width=True):
        st.session_state["diseases"] = all_diseases
    if col_d2.button("None", key="disease_none", use_container_width=True):
        st.session_state["diseases"] = []

    selected_diseases = st.multiselect(
        "Primary diseases",
        all_diseases,
        default=st.session_state.get("diseases", all_diseases),
        label_visibility="collapsed",
        key="diseases",
    )

    st.markdown("---")

    # ── Model ID selector ─────────────────────────────────────────────
    DEFAULT_MODEL_IDS = ["ACH-000323", "ACH-002083", "ACH-002228"]
    all_model_ids = list(reactome_matrix["ModelID"].unique())

    st.markdown(
        "<span style=\"font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:0.5px;\">ModelIDs (spider plots)</span>",
        unsafe_allow_html=True,
    )
    col_m1, col_m2, col_m3 = st.columns(3)
    if col_m1.button("All", key="model_all", use_container_width=True):
        st.session_state["model_ids"] = all_model_ids
    if col_m2.button("None", key="model_none", use_container_width=True):
        st.session_state["model_ids"] = []
    if col_m3.button("Default", key="model_default", use_container_width=True):
        st.session_state["model_ids"] = DEFAULT_MODEL_IDS

    model_ids = st.multiselect(
        "ModelIds (for spider plots)",
        all_model_ids,
        default=st.session_state.get("model_ids", DEFAULT_MODEL_IDS),
        label_visibility="collapsed",
        key="model_ids",
    )

    st.markdown("---")

    # ── Spider plot process limit ─────────────────────────────────────
    st.markdown(
        "<span style=\"font-size:11px;color:#8b949e;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:0.5px;\">Spider plot — max processes</span>",
        unsafe_allow_html=True,
    )

    limit_processes = st.toggle("Limit axes", value=False, key="limit_processes")

    if limit_processes:
        max_processes = st.slider(
            "Max processes per plot",
            min_value=3,
            max_value=20,
            value=20,
            step=1,
            label_visibility="collapsed",
        )
    else:
        max_processes = None  # signals downstream: show all

    st.markdown("---")
    st.markdown(
        '<span style="font-size:10px;color:#484f58;">Data: DepMap synthetic (mirrors WayScience/gene-process-dependencies schema)</span>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Gene Process Dependency Explorer")
st.markdown(
    '<span class="section-tag">DepMap · BioBombe · WayScience</span>',
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_latent, tab_spider, tab_scores, tab_table = st.tabs(
    [
        "📍 single dependency plots",
        "📍 latent plots",
        "🕷 Spider Plots",
        "📊 Top Scores",
        "📋 Data Table",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PCA
# ─────────────────────────────────────────────────────────────────────────────
with tab_latent:
    st.markdown("### Latent Space PCA Projections")
    st.markdown(
        "PCA projection of latent representations learned by the selected compression model. "
        "Each point is a cancer cell line; clusters indicate similar gene dependency profiles."
    )

    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
    # add a way to show all diseases or select specific ones

    reactome_matrix = reactome_matrix[
        reactome_matrix["OncotreePrimaryDisease"].isin(selected_diseases)
    ]
    corum_matrix = corum_matrix[
        corum_matrix["OncotreePrimaryDisease"].isin(selected_diseases)
    ]
    drug_matrix = drug_matrix[
        drug_matrix["OncotreePrimaryDisease"].isin(selected_diseases)
    ]
    # Assuming combined_df is your full dataset
    reactome_fig, reactome_out = make_dropdown_pca_with_selection(
        reactome_matrix, "PCA: Reactome Subset"
    )
    corum_fig, corum_out = make_dropdown_pca_with_selection(
        corum_matrix, "PCA: CORUM Subset"
    )
    drug_fig, drug_out = make_dropdown_pca_with_selection(
        drug_matrix, "PCA: Drug Subset"
    )
    st.plotly_chart(reactome_fig, use_container_width=True)
    st.plotly_chart(corum_fig, use_container_width=True)
    st.plotly_chart(drug_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PCA single
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Latent Space PCA Projections")
    st.markdown(
        "PCA projection of latent representations learned by the selected compression model. "
        "Each point is a cancer cell line; clusters indicate similar gene dependency profiles."
    )
    # st.markdown("**(Placeholder for latent space PCA plots)**")

    combined_df = single_load_data()
    combined_df = combined_df[
        combined_df["OncotreePrimaryDisease"].isin(selected_diseases)
    ]
    gene_cols = combined_df.columns.drop("ModelID")
    gene_cols = gene_cols.drop("OncotreePrimaryDisease")

    # Prepare PCA input (excluding non-numeric columns)
    pca_input = combined_df[gene_cols].apply(pd.to_numeric, errors="coerce")
    pca = PCA(n_components=2, random_state=0)
    pca_embedding = pca.fit_transform(pca_input)

    # Add PCA components to the dataframe
    combined_df["PCA1"] = pca_embedding[:, 0]
    combined_df["PCA2"] = pca_embedding[:, 1]

    # Prepare color map for each cancer type
    cancer_types = combined_df["OncotreePrimaryDisease"].unique()
    color_map = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Dark24
    )
    highlight_color_map = {
        cancer: color_map[i % len(color_map)] for i, cancer in enumerate(cancer_types)
    }
    default_colors = [highlight_color_map[cancer] for cancer in cancer_types]

    # Create one trace per cancer type
    traces = []
    for cancer in cancer_types:
        df_subset = combined_df[combined_df["OncotreePrimaryDisease"] == cancer]
        trace = go.Scatter(
            x=df_subset["PCA1"],
            y=df_subset["PCA2"],
            mode="markers",
            name=cancer,
            marker=dict(size=7),
            text=[f"{cancer} | {model_id}" for model_id in df_subset["ModelID"]],
            hoverinfo="text",
        )
        traces.append(trace)
    import plotly.graph_objects as go
    import streamlit as st

    # Create the figure
    fig = go.Figure(data=traces)

    # Set default colors
    for i, trace in enumerate(fig.data):
        trace.marker.color = default_colors[i]

    st.plotly_chart(fig, use_container_width=True)
    # fig.write_html("achilles_pca.html")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Spider / Radar plots
# ─────────────────────────────────────────────────────────────────────────────
with tab_spider:
    st.markdown("### Biological Process Dependency Profiles — Spider Plots")
    st.markdown(
        "Mean dependency score per biological process category for selected diseases. "
        "Negative scores indicate stronger dependencies (essential genes). "
        "Each axis = one process; the profile shape captures cancer-type-specific vulnerabilities."
    )

    dfs, global_max, feature_colnames = spider_load_data()

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), subplot_kw=dict(polar=True))

    for ax, (title, df) in zip(axes, dfs.items()):
        df = df[df["ModelID"].isin(model_ids)].copy()

        # Apply process limit if set
        cols = feature_colnames[title]

        make_radar(
            ax,
            df,
            cols,
            title,
            global_max,
            model_ids=model_ids,
            max_processes=max_processes,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Top Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab_scores:
    st.markdown("### Top Dependency Scores")
    st.markdown(
        "Visualize the highest-scoring biological processes across cancer types and individual cell lines. "
        "Use the controls below to tune how many processes, cancer types, and model IDs are shown."
    )

    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
    all_matrices = {
        "Reactome": reactome_matrix,
        "CORUM": corum_matrix,
        "Drug": drug_matrix,
    }

    # ── Controls ──────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        scores_dataset = st.selectbox(
            "Dataset",
            options=["Reactome", "CORUM", "Drug"],
            key="scores_dataset",
        )
    with sc2:
        top_n_scores = st.slider(
            "Top N processes",
            min_value=3,
            max_value=50,
            value=15,
            step=1,
            key="scores_top_n",
        )
    with sc3:
        top_m_types = st.slider(
            "Top M cancer types",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            key="scores_top_m",
        )
    with sc4:
        top_i_models = st.slider(
            "Top I model IDs",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            key="scores_top_i",
        )

    score_df = all_matrices[scores_dataset].copy()
    score_df = score_df[score_df["OncotreePrimaryDisease"].isin(selected_diseases)]

    meta_cols = ["ModelID", "OncotreePrimaryDisease"]
    score_cols = [c for c in score_df.columns if c not in meta_cols]

    # ── Plot 1: top N processes × top M cancer types ──────────────────
    st.markdown("---")
    st.markdown("#### By cancer type")
    st.markdown(
        "Mean dependency score per process, averaged across all cell lines in each cancer type. "
        "Only the top N processes (by max mean score across shown cancer types) are displayed."
    )

    # Mean score per (disease, process)
    disease_means = (
        score_df.groupby("OncotreePrimaryDisease")[score_cols].mean().reset_index()
    )

    # Pick top M cancer types by their single highest mean score across all processes
    disease_means["_max"] = disease_means[score_cols].max(axis=1)
    top_diseases = disease_means.nlargest(top_m_types, "_max")[
        "OncotreePrimaryDisease"
    ].tolist()
    disease_means = disease_means[
        disease_means["OncotreePrimaryDisease"].isin(top_diseases)
    ]
    disease_means = disease_means.drop(columns=["_max"])

    # Pick top N processes by max mean score across selected diseases
    process_maxes = disease_means[score_cols].max(axis=0)
    top_processes = process_maxes.nlargest(top_n_scores).index.tolist()

    heatmap_disease = disease_means.set_index("OncotreePrimaryDisease")[top_processes]

    fig_disease = go.Figure(
        go.Heatmap(
            z=heatmap_disease.values,
            x=[
                textwrap.shorten(p, width=30, placeholder="…")
                for p in heatmap_disease.columns
            ],
            y=heatmap_disease.index.tolist(),
            colorscale="Blues",
            hovertemplate="Disease: %{y}<br>Process: %{x}<br>Score: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Mean score", thickness=14, len=0.6),
        )
    )
    fig_disease.update_layout(
        height=max(300, top_m_types * 36 + 120),
        margin=dict(l=20, r=20, t=30, b=120),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        xaxis=dict(tickangle=-40, gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_disease, use_container_width=True)

    # ── Plot 2: top N processes × top I model IDs ─────────────────────
    st.markdown("---")
    st.markdown("#### By model ID")
    st.markdown(
        "Raw dependency score per process for individual cell lines. "
        "Only the top I model IDs (by their single highest score across all processes) are shown."
    )

    # Pick top I models by max score across all processes
    score_df["_max"] = score_df[score_cols].max(axis=1)
    top_models = score_df.nlargest(top_i_models, "_max")["ModelID"].tolist()
    score_df = score_df.drop(columns=["_max"])

    model_df = score_df[score_df["ModelID"].isin(top_models)].set_index("ModelID")[
        top_processes
    ]

    fig_model = go.Figure(
        go.Heatmap(
            z=model_df.values,
            x=[
                textwrap.shorten(p, width=30, placeholder="…") for p in model_df.columns
            ],
            y=model_df.index.tolist(),
            colorscale="Purples",
            hovertemplate="Model: %{y}<br>Process: %{x}<br>Score: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Score", thickness=14, len=0.6),
        )
    )
    fig_model.update_layout(
        height=max(300, top_i_models * 36 + 120),
        margin=dict(l=20, r=20, t=30, b=120),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        xaxis=dict(tickangle=-40, gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_model, use_container_width=True)

    # ── Bar charts: top N processes collapsed ─────────────────────────
    st.markdown("---")
    st.markdown("#### Overall top processes")
    st.markdown(
        "Global mean score per process across all cell lines in the filtered dataset, ranked."
    )

    overall_means = score_df[score_cols].mean().nlargest(top_n_scores).reset_index()
    overall_means.columns = ["Process", "Mean Score"]
    overall_means["Process"] = overall_means["Process"].apply(
        lambda p: textwrap.shorten(p, width=40, placeholder="…")
    )

    bar1, bar2 = st.columns(2)

    with bar1:
        fig_bar_d = px.bar(
            overall_means,
            x="Mean Score",
            y="Process",
            orientation="h",
            title="Top processes — all cell lines",
            color="Mean Score",
            color_continuous_scale="Blues",
        )
        fig_bar_d.update_layout(
            height=max(300, top_n_scores * 24 + 80),
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", size=11),
            yaxis=dict(autorange="reversed", gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False,
            showlegend=False,
        )
        st.plotly_chart(fig_bar_d, use_container_width=True)

    with bar2:
        # Same but restricted to the top-M diseases
        top_disease_means = (
            score_df[score_df["OncotreePrimaryDisease"].isin(top_diseases)][score_cols]
            .mean()
            .nlargest(top_n_scores)
            .reset_index()
        )
        top_disease_means.columns = ["Process", "Mean Score"]
        top_disease_means["Process"] = top_disease_means["Process"].apply(
            lambda p: textwrap.shorten(p, width=40, placeholder="…")
        )
        fig_bar_m = px.bar(
            top_disease_means,
            x="Mean Score",
            y="Process",
            orientation="h",
            title=f"Top processes — top {top_m_types} cancer types",
            color="Mean Score",
            color_continuous_scale="Purples",
        )
        fig_bar_m.update_layout(
            height=max(300, top_n_scores * 24 + 80),
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", size=11),
            yaxis=dict(autorange="reversed", gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False,
            showlegend=False,
        )
        st.plotly_chart(fig_bar_m, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Data Table
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.markdown("### Cancer Types & Primary Diseases")
    st.markdown(
        "Summary of all cancer types and their associated primary diseases present in the dataset."
    )

    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
    full_df = reactome_matrix[
        reactome_matrix["OncotreePrimaryDisease"].isin(selected_diseases)
    ]

    # Build summary: one row per unique (OncotreeLineage, OncotreePrimaryDisease) pair
    # adjust column names below to match whatever lineage/cancer-type column your data actually has
    group_cols = ["OncotreePrimaryDisease", "ModelID"]  # ← rename if needed
    summary = (
        reactome_matrix.groupby(group_cols, dropna=False)
        .agg(n_models=("ModelID", "nunique"))
        .reset_index()
        .sort_values(group_cols)
        .rename(
            columns={
                "OncotreePrimaryDisease": "Primary Disease",
                "n_models": "Cell Lines",
            }
        )
        .drop(columns=["Cell Lines"])
        .drop_duplicates()
    )

    # ── Summary metrics ───────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.markdown(
        f'<div class="metric-card"><div class="metric-label">Primary diseases</div><div class="metric-value">{summary["Primary Disease"].nunique()}</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><div class="metric-label">Total cell lines</div><div class="metric-value">{int(summary["ModelID"].nunique())}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Sort controls ─────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        sort_by = st.selectbox(
            "Sort by",
            options=summary.columns.tolist(),
            index=0,
            key="table_sort_col",
        )
    with sc2:
        sort_asc = st.radio(
            "Order",
            ["Ascending", "Descending"],
            index=0,
            horizontal=True,
            key="table_sort_dir",
        )

    summary = summary.sort_values(sort_by, ascending=(sort_asc == "Ascending"))

    # ── Table ─────────────────────────────────────────────────────────
    st.dataframe(
        summary.reset_index(drop=True),
        use_container_width=True,
        height=560,
        column_config={
            "Lineage / Cancer Type": st.column_config.TextColumn(width="large"),
            "Primary Disease": st.column_config.TextColumn(width="large"),
            "Cell Lines": st.column_config.NumberColumn(width="small", format="%d"),
        },
        hide_index=True,
    )

    # ── Download ──────────────────────────────────────────────────────
    st.download_button(
        label="⬇ Download as CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="cancer_types_diseases.csv",
        mime="text/csv",
        key="table_download",
    )
