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
    st.markdown("## 🧬 Controls")
    st.markdown("---")
    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
    # st.markdown("**DISEASE TYPES**")
    # selected_diseases = st.multiselect(
    #     "Primary diseases",
    #     PRIMARY_DISEASES,
    #     default=PRIMARY_DISEASES[:8],
    #     label_visibility="collapsed",
    # )
    selected_diseases = st.multiselect(
        "Primary diseases",
        reactome_matrix["OncotreePrimaryDisease"].unique(),
        default=reactome_matrix["OncotreePrimaryDisease"].unique(),
        label_visibility="collapsed",
    )

    model_ids = st.multiselect(
        "ModelIds (for spider plots)",
        ["ACH-000323", "ACH-002083", "ACH-002228"],
        default=["ACH-000323", "ACH-002083", "ACH-002228"],
        label_visibility="collapsed",
    )

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
# tab_single = st.tabs(
# tab_single, tab_latent= st.tabs(
(tab_spider,) = st.tabs(["🕷 Spider Plots"])
# tab_single, = st.tabs(["📍 single dependency plots"])
# tab_latent, = st.tabs(["📍 latent plots"])
# tab_single, tab_latent, tab_spider = st.tabs(
# ["📍 single dependency plots", "📍 latent plots", "🕷 Spider Plots"]
# ["📍 single dependency plots", "📍 latent plots", "🕷 Spider Plots"]


# # ─────────────────────────────────────────────────────────────────────────────
# # TAB 1: PCA
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_latent:
#     st.markdown("### Latent Space PCA Projections")
#     st.markdown(
#         "PCA projection of latent representations learned by the selected compression model. "
#         "Each point is a cancer cell line; clusters indicate similar gene dependency profiles."
#     )


#     reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
#     # add a way to show all diseases or select specific ones

#     reactome_matrix = reactome_matrix[reactome_matrix["OncotreePrimaryDisease"].isin(selected_diseases)]
#     corum_matrix = corum_matrix[corum_matrix["OncotreePrimaryDisease"].isin(selected_diseases)]
#     drug_matrix = drug_matrix[drug_matrix["OncotreePrimaryDisease"].isin(selected_diseases)]
#     # Assuming combined_df is your full dataset
#     reactome_fig, reactome_out = make_dropdown_pca_with_selection(reactome_matrix, "PCA: Reactome Subset")
#     corum_fig, corum_out = make_dropdown_pca_with_selection(corum_matrix, "PCA: CORUM Subset")
#     drug_fig, drug_out = make_dropdown_pca_with_selection(drug_matrix, "PCA: Drug Subset")
#     st.plotly_chart(reactome_fig, use_container_width=True)
#     st.plotly_chart(corum_fig, use_container_width=True)
#     st.plotly_chart(drug_fig, use_container_width=True)


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

    # --- Create one combined figure ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), subplot_kw=dict(polar=True))

    for ax, (title, df) in zip(axes, dfs.items()):
        df = df[df["ModelID"].isin(model_ids)].copy()
        make_radar(ax, df, feature_colnames[title], title, global_max)

    # Shared legend (one for all)
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
