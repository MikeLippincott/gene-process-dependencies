"""
Gene Process Dependency Explorer
"""

import os
import pathlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app_utils import (
    clean_label,
    compute_single_pca,
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

BASE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = BASE_DIR.parent
print(f"Base directory: {BASE_DIR}")
print(f"Repository directory: {REPO_DIR}")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gene Process Dependency Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
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

  .control-panel {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }
  .control-label {
    font-size: 11px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }

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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Gene Process Dependency Explorer")
st.markdown(
    '<span class="section-tag">DepMap · BioBombe · WayScience</span>',
    unsafe_allow_html=True,
)

# ── Load only the tiny model metadata file at startup (188 KB) ────────────────
_model_meta = pd.read_parquet(
    BASE_DIR / "data" / "Model.parquet",
    columns=["ModelID", "OncotreePrimaryDisease"],
)
ALL_DISEASES = sorted(_model_meta["OncotreePrimaryDisease"].dropna().unique().tolist())
ALL_MODEL_IDS = sorted(_model_meta["ModelID"].dropna().unique().tolist())
DEFAULT_MODEL_IDS = ["ACH-000323", "ACH-002083", "ACH-002228"]
single_gene_pca_df = pd.read_parquet(
    BASE_DIR / "data" / "pca_embeddings_single_dependencies.parquet"
)
latent_reactome_pca_df = pd.read_parquet(
    BASE_DIR / "data" / "pca_embeddings_latent_reactome.parquet"
)
latent_corum_pca_df = pd.read_parquet(
    BASE_DIR / "data" / "pca_embeddings_latent_corum.parquet"
)
latent_drug_pca_df = pd.read_parquet(
    BASE_DIR / "data" / "pca_embeddings_latent_drug.parquet"
)
# ── Tabs ──────────────────────────────────────────────────────────────────────
(
    tab_welcome,
    tab_single,
    tab_latent,
    tab_spider,
    tab_scores,
    tab_table,
) = st.tabs(
    [
        "📖 Welcome",
        "📍 Single dependency plots",
        "📍 Latent plots",
        "🕷 Spider plots",
        "📊 Top scores",
        "📋 Data table",
    ]
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def disease_controls(key_prefix: str, default_all: bool = True) -> list[str]:
    """Render an All / None button pair + multiselect for disease filtering."""
    st.markdown(
        '<div class="control-label">Primary diseases</div>', unsafe_allow_html=True
    )
    ca, cn = st.columns(2)
    if ca.button("All", key=f"{key_prefix}_dis_all", use_container_width=True):
        st.session_state[f"{key_prefix}_diseases"] = ALL_DISEASES
    if cn.button("None", key=f"{key_prefix}_dis_none", use_container_width=True):
        st.session_state[f"{key_prefix}_diseases"] = []
    return st.multiselect(
        "Primary diseases",
        ALL_DISEASES,
        default=st.session_state.get(
            f"{key_prefix}_diseases", ALL_DISEASES if default_all else []
        ),
        label_visibility="collapsed",
        key=f"{key_prefix}_diseases",
    )


def model_id_controls(key_prefix: str) -> list[str]:
    """Render All / None / Default buttons + multiselect for ModelID filtering."""
    st.markdown('<div class="control-label">Model IDs</div>', unsafe_allow_html=True)
    ca, cn, cd = st.columns(3)
    if ca.button("All", key=f"{key_prefix}_mod_all", use_container_width=True):
        st.session_state[f"{key_prefix}_model_ids"] = ALL_MODEL_IDS
    if cn.button("None", key=f"{key_prefix}_mod_none", use_container_width=True):
        st.session_state[f"{key_prefix}_model_ids"] = []
    if cd.button("Default", key=f"{key_prefix}_mod_default", use_container_width=True):
        st.session_state[f"{key_prefix}_model_ids"] = DEFAULT_MODEL_IDS
    return st.multiselect(
        "Model IDs",
        ALL_MODEL_IDS,
        default=st.session_state.get(f"{key_prefix}_model_ids", DEFAULT_MODEL_IDS),
        label_visibility="collapsed",
        key=f"{key_prefix}_model_ids",
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 0 — Intro text
# ────────────────────────────────────────────────────────────────────────────
with tab_welcome:
    st.markdown("## Welcome to the Gene Process Dependency Explorer")
    st.markdown(
        "This app allows you to explore gene process dependencies across different cancer types and model IDs."
    )
    st.markdown(
        "We are visualizing latent representations of gene dependencies learned by a compression model trained on DepMap data, with biological processes as features."
    )
    st.markdown(
        "For more details on the data and methods, please see our github repository: https://github.com/WayScience/gene-process-dependencies"
    )
    st.markdown(
        "For more information about the group, please visit the WayScience website: https://wayscience.com"
    )
    st.markdown("---")
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single dependency plots
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Latent Space PCA — single dependency")
    st.markdown(
        "PCA projection of raw gene dependency scores. "
        "Each point is a cancer cell line; colour encodes primary disease."
    )

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        selected_diseases_single = disease_controls("single")
        st.markdown("</div>", unsafe_allow_html=True)

    cancer_types = single_gene_pca_df["OncotreePrimaryDisease"].unique()
    color_map = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Dark24
    )
    highlight_color_map = {
        c: color_map[i % len(color_map)] for i, c in enumerate(cancer_types)
    }

    traces = []
    for cancer in cancer_types:
        df_sub = single_gene_pca_df[
            single_gene_pca_df["OncotreePrimaryDisease"] == cancer
        ]
        is_selected = cancer in selected_diseases_single
        traces.append(
            go.Scatter(
                x=df_sub["PCA1"],
                y=df_sub["PCA2"],
                mode="markers",
                name=cancer,
                marker=dict(
                    size=7,
                    color=highlight_color_map[cancer] if is_selected else "#3d444d",
                    opacity=1.0 if is_selected else 0.3,
                ),
                text=[f"{cancer} | {m}" for m in df_sub["ModelID"]],
                hoverinfo="text" if is_selected else "skip",
                showlegend=is_selected,
            )
        )

    # Render grayed traces behind colored ones
    traces.sort(key=lambda t: t.name in selected_diseases_single, reverse=True)

    fig_single = go.Figure(data=traces)
    fig_single.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=1200,
        margin=dict(l=40, r=40, t=40, b=40),
        width=800,
    )
    st.plotly_chart(fig_single, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Latent plots
# ─────────────────────────────────────────────────────────────────────────────
with tab_latent:
    st.markdown("### Latent Space PCA Projections")
    st.markdown(
        "PCA projection of latent representations learned by the selected compression model. "
        "Each point is a cancer cell line; clusters indicate similar gene dependency profiles."
    )

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        selected_diseases_latent = disease_controls("latent")
        st.markdown("</div>", unsafe_allow_html=True)

    for title, pca_df in [
        ("Reactome", latent_reactome_pca_df),
        ("CORUM", latent_corum_pca_df),
        ("Drug", latent_drug_pca_df),
    ]:
        st.markdown(f"#### PCA: {title} Subset")

        cancer_types = pca_df["OncotreePrimaryDisease"].unique()
        color_map = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.Light24
            + px.colors.qualitative.Dark24
        )
        highlight_color_map = {
            c: color_map[i % len(color_map)] for i, c in enumerate(cancer_types)
        }

        traces = []
        for cancer in cancer_types:
            df_sub = pca_df[pca_df["OncotreePrimaryDisease"] == cancer]
            is_selected = cancer in selected_diseases_latent
            traces.append(
                go.Scatter(
                    x=df_sub["PCA1"],
                    y=df_sub["PCA2"],
                    mode="markers",
                    name=cancer,
                    marker=dict(
                        size=7,
                        color=highlight_color_map[cancer] if is_selected else "#3d444d",
                        opacity=1.0 if is_selected else 0.3,
                    ),
                    text=[f"{cancer} | {m}" for m in df_sub["ModelID"]],
                    hoverinfo="text" if is_selected else "skip",
                    showlegend=is_selected,
                )
            )

        traces.sort(key=lambda t: t.name in selected_diseases_latent, reverse=True)

        fig = go.Figure(data=traces)
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            height=600,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Spider plots
# ─────────────────────────────────────────────────────────────────────────────
with tab_spider:
    st.markdown("### Biological Process Dependency Profiles — Spider Plots")
    st.markdown(
        "Mean dependency score per biological process category for selected cell lines. "
        "Negative scores indicate stronger dependencies (essential genes)."
    )

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        sp_c1, sp_c2, sp_c3 = st.columns(3)

        with sp_c1:
            spider_filter_mode = st.radio(
                "Filter by",
                ["Model IDs", "Cancer Types"],
                horizontal=True,
                key="spider_filter_mode",
            )

        with sp_c2:
            if spider_filter_mode == "Model IDs":
                st.markdown(
                    '<div class="control-label">Model IDs (max 10)</div>',
                    unsafe_allow_html=True,
                )
                spider_model_ids = st.multiselect(
                    "Model IDs",
                    ALL_MODEL_IDS,
                    default=st.session_state.get("spider_model_ids", DEFAULT_MODEL_IDS),
                    max_selections=10,
                    label_visibility="collapsed",
                    key="spider_model_ids",
                )
            else:
                st.markdown(
                    '<div class="control-label">Cancer Types (max 5)</div>',
                    unsafe_allow_html=True,
                )
                spider_cancer_types = st.multiselect(
                    "Cancer Types",
                    ALL_DISEASES,
                    default=st.session_state.get(
                        "spider_cancer_types", ALL_DISEASES[:2]
                    ),
                    max_selections=5,
                    label_visibility="collapsed",
                    key="spider_cancer_types",
                )
                st.markdown(
                    '<div class="control-label">Aggregation</div>',
                    unsafe_allow_html=True,
                )
                spider_agg = st.radio(
                    "Aggregation",
                    ["Mean", "Min", "Max"],
                    horizontal=True,
                    key="spider_agg",
                    label_visibility="collapsed",
                )

        with sp_c3:
            st.markdown(
                '<div class="control-label">Process limit</div>', unsafe_allow_html=True
            )
            limit_processes = st.toggle(
                "Limit axes", value=False, key="spider_limit_toggle"
            )
            if limit_processes:
                max_processes = st.slider(
                    "Max processes per plot",
                    min_value=3,
                    max_value=50,
                    value=20,
                    step=1,
                    key="spider_max_proc",
                    label_visibility="collapsed",
                )
            else:
                max_processes = None
        st.markdown("</div>", unsafe_allow_html=True)

    dfs, global_max, feature_colnames = spider_load_data()

    if spider_filter_mode == "Model IDs":
        selected_ids = spider_model_ids
        id_col = "ModelID"
    else:
        selected_ids = (
            spider_cancer_types
            if "spider_cancer_types" in st.session_state or True
            else []
        )
        id_col = "OncotreePrimaryDisease"
        agg_func = {"Mean": "mean", "Min": "min", "Max": "max"}[spider_agg]

    if len(selected_ids) == 0:
        st.warning(
            "Please select at least one model ID or cancer type to display the plots."
        )
    else:
        fig_spider, axes = plt.subplots(
            3, 1, figsize=(18, 12), subplot_kw=dict(polar=True)
        )

        for ax, (title, df) in zip(axes, dfs.items()):
            cols = feature_colnames[title]
            if spider_filter_mode == "Cancer Types":
                df_filtered = df[df["OncotreePrimaryDisease"].isin(selected_ids)].copy()

                # cols from feature_colnames[title] is the correct feature column list
                feature_cols = list(cols) if not isinstance(cols, list) else cols
                st.write("feature_colnames:", feature_colnames)

                df_agg = (
                    df_filtered.groupby("OncotreePrimaryDisease")[feature_cols]
                    .agg(agg_func)
                    .reset_index()
                    .rename(columns={"OncotreePrimaryDisease": "ModelID"})
                )

                # Melt to long format — use a fixed string for var_name
                df_long = df_agg.melt(
                    id_vars="ModelID",
                    value_vars=feature_cols,
                    var_name="feature",  # fixed string, not cols
                    value_name="agg_score",
                )

                make_radar(
                    ax,
                    df_long,
                    "feature",  # must match var_name above
                    f"{title} ({spider_agg})",
                    global_max,
                    score_col="agg_score",
                    model_ids=selected_ids,
                    max_processes=max_processes,
                )
            else:
                df_filtered = df[df["ModelID"].isin(selected_ids)].copy()
                make_radar(
                    ax,
                    df_filtered,
                    cols,
                    title,
                    global_max,
                    model_ids=selected_ids,
                    max_processes=max_processes,
                )

        handles, labels = axes[0].get_legend_handles_labels()
        fig_spider.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig_spider)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Top scores
# ─────────────────────────────────────────────────────────────────────────────
with tab_scores:
    st.markdown("### Top Dependency Scores")
    st.markdown(
        "Highest-scoring biological processes across cancer types and individual cell lines."
    )

    reactome_matrix, corum_matrix, drug_matrix = latent_load_data()
    all_matrices = {
        "Reactome": reactome_matrix,
        "CORUM": corum_matrix,
        "Drug": drug_matrix,
    }

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns(5)
        with ctl1:
            selected_diseases_scores = disease_controls("scores")
        with ctl2:
            scores_dataset = st.selectbox(
                "Dataset",
                options=["Reactome", "CORUM", "Drug"],
                key="scores_dataset",
            )
        with ctl3:
            top_n_scores = st.slider(
                "Top N processes",
                min_value=3,
                max_value=50,
                value=15,
                step=1,
                key="scores_top_n",
            )
        with ctl4:
            top_m_types = st.slider(
                "Top M cancer types",
                min_value=1,
                max_value=len(selected_diseases_scores),
                value=8,
                step=1,
                key="scores_top_m",
            )
        with ctl5:
            top_i_models = st.slider(
                "Top I model IDs",
                min_value=1,
                max_value=30,
                value=8,
                step=1,
                key="scores_top_i",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    score_df = all_matrices[scores_dataset].copy()
    score_df = score_df[
        score_df["OncotreePrimaryDisease"].isin(selected_diseases_scores)
    ]
    meta_cols = ["ModelID", "OncotreePrimaryDisease"]
    score_cols = [c for c in score_df.columns if c not in meta_cols]

    # ── Heatmap: cancer types ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### By cancer type")
    st.markdown(
        "Mean dependency score per process averaged across all cell lines in each cancer type. "
        "Top N processes by max mean score across shown cancer types."
    )

    disease_means = (
        score_df.groupby("OncotreePrimaryDisease")[score_cols].mean().reset_index()
    )
    disease_means["_max"] = disease_means[score_cols].max(axis=1)
    top_diseases = disease_means.nlargest(top_m_types, "_max")[
        "OncotreePrimaryDisease"
    ].tolist()
    disease_means = disease_means[
        disease_means["OncotreePrimaryDisease"].isin(top_diseases)
    ].drop(columns=["_max"])
    top_processes = (
        disease_means[score_cols].max(axis=0).nlargest(top_n_scores).index.tolist()
    )
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

    # ── Heatmap: model IDs ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### By model ID")
    st.markdown(
        "Raw dependency score per process for individual cell lines ranked by peak score."
    )

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

    # ── Bar charts ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Overall top processes")
    st.markdown("Global mean score per process across all filtered cell lines, ranked.")

    overall_means = score_df[score_cols].mean().nlargest(top_n_scores).reset_index()
    overall_means.columns = ["Process", "Mean Score"]
    overall_means["Process"] = overall_means["Process"].apply(
        lambda p: textwrap.shorten(p, width=40, placeholder="…")
    )

    bar1, bar2 = st.columns(2)

    _bar_layout = dict(
        margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        yaxis=dict(autorange="reversed", gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
        coloraxis_showscale=False,
        showlegend=False,
    )

    with bar1:
        fig_bar_all = px.bar(
            overall_means,
            x="Mean Score",
            y="Process",
            orientation="h",
            title="Top processes — all cell lines",
            color="Mean Score",
            color_continuous_scale="Blues",
        )
        fig_bar_all.update_layout(
            height=max(300, top_n_scores * 24 + 80), **_bar_layout
        )
        st.plotly_chart(fig_bar_all, use_container_width=True)

    with bar2:
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
        fig_bar_top = px.bar(
            top_disease_means,
            x="Mean Score",
            y="Process",
            orientation="h",
            title=f"Top processes — top {top_m_types} cancer types",
            color="Mean Score",
            color_continuous_scale="Purples",
        )
        fig_bar_top.update_layout(
            height=max(300, top_n_scores * 24 + 80), **_bar_layout
        )
        st.plotly_chart(fig_bar_top, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Data table
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.markdown("### Cancer Types & Primary Diseases")
    st.markdown("Summary of all cancer types and their associated primary diseases.")

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        selected_diseases_table = disease_controls("table")
        st.markdown("</div>", unsafe_allow_html=True)

    reactome_matrix, _, _ = latent_load_data()
    full_df = reactome_matrix[
        reactome_matrix["OncotreePrimaryDisease"].isin(selected_diseases_table)
    ]

    summary = (
        full_df.groupby(["OncotreePrimaryDisease", "ModelID"], dropna=False)
        .size()
        .reset_index(name="_n")
        .drop(columns=["_n"])
        .drop_duplicates()
        .rename(columns={"OncotreePrimaryDisease": "Primary Disease"})
        .sort_values(["Primary Disease", "ModelID"])
    )

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

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        sort_by = st.selectbox(
            "Sort by", options=summary.columns.tolist(), index=0, key="table_sort_col"
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

    st.dataframe(
        summary.reset_index(drop=True),
        use_container_width=True,
        height=560,
        column_config={
            "Primary Disease": st.column_config.TextColumn(width="large"),
            "ModelID": st.column_config.TextColumn("Model ID", width="large"),
        },
        hide_index=True,
    )

    st.download_button(
        label="⬇ Download as CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="cancer_types_diseases.csv",
        mime="text/csv",
        key="table_download",
    )
