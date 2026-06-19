import colorsys
import pathlib
import random
import textwrap

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def latent_load_data():
    reactome_dims = DATA_DIR / "all_reactome_results.parquet"
    corum_dims = DATA_DIR / "all_corum_results.parquet"
    drug_dims = DATA_DIR / "all_drug_results.parquet"

    reactome_df = pd.read_parquet(reactome_dims)
    corum_df = pd.read_parquet(corum_dims)
    drug_df = pd.read_parquet(drug_dims)

    reactome_df["feature"] = reactome_df["reactome_pathway"]
    reactome_df = reactome_df.drop(
        columns=["model", "latent_dim_total", "init", "seed", "z", "reactome_pathway"]
    )

    corum_df["feature"] = corum_df["reactome_pathway"]
    corum_df = corum_df.drop(
        columns=["model", "latent_dim_total", "init", "seed", "z", "reactome_pathway"]
    )

    drug_df["feature"] = drug_df["name"]
    drug_df = drug_df.drop(
        columns=["model", "latent_dim_total", "init", "seed", "z", "name"]
    )

    meta_cols = ["ModelID", "OncotreePrimaryDisease"]
    reactome_meta = reactome_df[meta_cols].drop_duplicates()
    corum_meta = corum_df[meta_cols].drop_duplicates()
    drug_meta = drug_df[meta_cols].drop_duplicates()

    reactome_matrix = reactome_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )
    corum_matrix = corum_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )
    drug_matrix = drug_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )

    reactome_matrix = reactome_matrix.merge(reactome_meta, on="ModelID", how="left")
    corum_matrix = corum_matrix.merge(corum_meta, on="ModelID", how="left")
    drug_matrix = drug_matrix.merge(drug_meta, on="ModelID", how="left")

    return reactome_matrix, corum_matrix, drug_matrix


@st.cache_data
def spider_load_data():
    files = {
        "Reactome Pathways": DATA_DIR / "all_reactome_results.parquet",
        "CORUM Complexes": DATA_DIR / "all_corum_results.parquet",
        "Drug Responses": DATA_DIR / "all_drug_results.parquet",
    }
    feature_colnames = {
        "Reactome Pathways": "reactome_pathway",
        "CORUM Complexes": "reactome_pathway",
        "Drug Responses": "name",
    }
    score_col = "pathway_score"

    global_max = 0
    dfs = {}
    for name, file in files.items():
        df = pd.read_parquet(file)
        dfs[name] = df
        if df[score_col].max() > global_max:
            global_max = df[score_col].max()

    return dfs, global_max, feature_colnames


def clean_label(x: str) -> str:
    if "Polymerase Switching" in x:
        return "Polymerase Switching"
    if "Negative Regulation Of MET" in x or "Negative Regulation of MET" in x:
        return "Negative MET Regulation"
    return truncate_label(x)


def truncate_label(
    label: str,
    max_chars_per_line: int = 22,
    max_total_words: int = 4,
    max_lines: int = 2,
) -> str:
    words = label.split()
    short = " ".join(words[:max_total_words])
    lines = textwrap.wrap(short, width=max_chars_per_line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    wrapped = "\n".join(lines)
    if len(words) > max_total_words or (
        len(lines) >= max_lines
        and len(lines) < len(textwrap.wrap(short, width=max_chars_per_line))
    ):
        wrapped += "..."
    return wrapped


def place_labels_polar(ax, angles, labels, radius: float) -> None:
    for angle, label in zip(angles[:-1], labels):
        if 0 <= angle <= np.pi / 2 or 3 * np.pi / 2 <= angle <= 2 * np.pi:
            ha = "left"
        else:
            ha = "right"
        ax.text(
            angle, radius, label, fontsize=9, color="black", ha=ha, va="center", wrap=True
        )


def generate_random_palette(num_colors, seed=12):
    random.seed(seed)
    colors = []
    for _ in range(num_colors):
        h = random.random()
        lightness = random.uniform(0.2, 0.8)
        s = random.uniform(0.5, 1.0)
        colors.append(colorsys.hls_to_rgb(h, lightness, s))
    return colors


def generate_model_color_map(model_ids, seed=12):
    colors = generate_random_palette(len(model_ids), seed)
    return {model_id: color for model_id, color in zip(model_ids, colors)}


def make_radar(
    ax,
    df: pd.DataFrame,
    feature_col: str,
    title: str,
    global_max: float,
    score_col: str = "pathway_score",
    model_ids: list = ["ACH-000323", "ACH-002083", "ACH-002228"],
    num_circles: int = 5,
    max_processes: int = None,
) -> None:
    if model_ids == ["ACH-000323", "ACH-002083", "ACH-002228"]:
        model_colors = {
            "ACH-000323": "#00FFFF",
            "ACH-002083": "#FF69B4",
            "ACH-002228": "#800080",
        }
    else:
        model_colors = generate_model_color_map(model_ids, seed=12)

    top_by_model = (
        df.groupby("ModelID", group_keys=False)
        .apply(lambda x: x.nlargest(5, score_col))
        .reset_index(drop=True)
    )
    top_features = top_by_model[feature_col].drop_duplicates().tolist()
    if max_processes is not None:
        top_features = top_features[:max_processes]
    n_vars = len(top_features)
    labels = [clean_label(x) for x in top_features]

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += [angles[0]]

    circle_levels = np.linspace(0, global_max, num_circles + 1)[1:]
    for r in circle_levels:
        ax.plot(
            angles, [r] * (n_vars + 1), color="grey", linestyle="--", linewidth=1, zorder=0
        )

    for mid in model_ids:
        subset = df[df["ModelID"] == mid]
        merged = pd.DataFrame({feature_col: top_features}).merge(
            subset[[feature_col, score_col]], on=feature_col, how="left"
        )
        values = merged[score_col].fillna(0).tolist()
        values += [values[0]]
        ax.plot(angles, values, color=model_colors[mid], linewidth=3, label=mid)
        ax.scatter(
            angles, values, color=model_colors[mid], s=30, edgecolor="black", zorder=5
        )

    ax.set_ylim(0, global_max * 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    place_labels_polar(ax, angles, labels, global_max * 1.4)
    ax.set_title(title, fontsize=18, pad=30, fontweight="bold", y=1)
