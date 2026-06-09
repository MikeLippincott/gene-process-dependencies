import colorsys
import pathlib
import random
import textwrap

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ipywidgets import widgets
from sklearn.decomposition import PCA


# ── Load & filter data ────────────────────────────────────────────────────────
def load_data():
    # Load dependency data
    data_directory = pathlib.Path("../0.data-download/data").resolve()
    dependency_file = pathlib.Path(
        f"{data_directory}/CRISPRGeneEffect.parquet"
    ).resolve()
    gene_dict_file = pathlib.Path(
        f"{data_directory}/CRISPR_gene_dictionary.parquet"
    ).resolve()
    dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)
    dependency_df = dependency_df.set_index("ModelID")
    cancer_type_input_file = pathlib.Path("../0.data-download/data/Model.parquet")
    cancer_type_df = pd.read_parquet(cancer_type_input_file)


def latent_load_data():
    # latent space data
    cancer_type_input_file = pathlib.Path("../0.data-download/data/Model.parquet")
    cancer_type_df = pd.read_parquet(cancer_type_input_file)

    reactome_dims = pathlib.Path(
        "../5.drug-dependency/results/all_reactome_results.parquet"
    )
    reactome_df = pd.read_parquet(reactome_dims)

    corum_dims = pathlib.Path("../5.drug-dependency/results/all_corum_results.parquet")
    corum_df = pd.read_parquet(corum_dims)

    drug_dims = pathlib.Path("../5.drug-dependency/results/all_drug_results.parquet")
    drug_df = pd.read_parquet(drug_dims)
    # Step 2: Subset based on matching keys
    subset_keys = ["model", "latent_dim_total", "init", "z"]

    # Replace latent identifiers with pathway
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
    drug_df_sorted = drug_df.sort_values(
        by=["OncotreePrimaryDisease", "ModelID", "feature"], ascending=True
    )
    # Create a unique column name by combining feature and source
    meta_cols = ["ModelID", "OncotreePrimaryDisease"]
    reactome_meta = reactome_df[meta_cols].drop_duplicates()
    corum_meta = corum_df[meta_cols].drop_duplicates()
    drug_meta = drug_df[meta_cols].drop_duplicates()

    # Pivot to wide format
    reactome_matrix = reactome_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )
    corum_matrix = corum_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )
    drug_matrix = drug_df.pivot(
        index="ModelID", columns="feature", values="latent_score"
    )

    # Join metadata back to each matrix
    reactome_matrix = reactome_matrix.merge(reactome_meta, on="ModelID", how="left")
    corum_matrix = corum_matrix.merge(corum_meta, on="ModelID", how="left")
    drug_matrix = drug_matrix.merge(drug_meta, on="ModelID", how="left")

    return reactome_matrix, corum_matrix, drug_matrix


def single_load_data():

    # Load dependency data
    data_directory = pathlib.Path("../0.data-download/data").resolve()
    dependency_file = pathlib.Path(
        f"{data_directory}/CRISPRGeneEffect.parquet"
    ).resolve()
    gene_dict_file = pathlib.Path(
        f"{data_directory}/CRISPR_gene_dictionary.parquet"
    ).resolve()
    dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)
    dependency_df = dependency_df.set_index("ModelID")
    cancer_type_input_file = pathlib.Path("../0.data-download/data/Model.parquet")
    cancer_type_df = pd.read_parquet(cancer_type_input_file)
    combined_df = dependency_df.merge(
        cancer_type_df[["ModelID", "OncotreePrimaryDisease"]], on="ModelID", how="left"
    )
    return combined_df


def spider_load_data():
    # model_ids = ["ACH-000323", "ACH-002083", "ACH-002228"]
    files = {
        "Reactome Pathways": "../5.drug-dependency/results/all_reactome_results.parquet",
        "CORUM Complexes": "../5.drug-dependency/results/all_corum_results.parquet",
        "Drug Responses": "../5.drug-dependency/results/all_drug_results.parquet",
    }
    # Adjust these if your column names differ
    feature_colnames = {
        "Reactome Pathways": "reactome_pathway",
        "CORUM Complexes": "reactome_pathway",
        "Drug Responses": "name",
    }
    score_col = "pathway_score"

    model_colors = {
        "ACH-000323": "#00FFFF",  # cyan
        "ACH-002083": "#FF69B4",  # pink
        "ACH-002228": "#800080",  # purple
    }

    global_max = 0
    dfs = {}
    for name, file in files.items():
        df = pd.read_parquet(file)
        dfs[name] = df
        if df[score_col].max() > global_max:
            global_max = df[score_col].max()

    return dfs, global_max, feature_colnames


def clean_label(x: str) -> str:
    """
    Clean or override specific long labels before truncation.
    """
    # Polymerase switching
    if "Polymerase Switching" in x:
        return "Polymerase Switching"

    # Negative MET regulation
    if "Negative Regulation Of MET" in x or "Negative Regulation of MET" in x:
        return "Negative MET Regulation"

    return truncate_label(x)


def truncate_label(
    label: str,
    max_chars_per_line: int = 22,
    max_total_words: int = 4,
    max_lines: int = 2,
) -> str:
    """
    Truncate a label according to character and line limits.
    """
    words = label.split()
    short = " ".join(words[:max_total_words])

    # Wrap
    lines = textwrap.wrap(short, width=max_chars_per_line)

    # clamp to max lines
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    wrapped = "\n".join(lines)

    # Determine if truncated
    if len(words) > max_total_words or (
        len(lines) >= max_lines
        and len(lines) < len(textwrap.wrap(short, width=max_chars_per_line))
    ):
        wrapped += "..."

    return wrapped


def place_labels_polar(ax, angles, labels, radius: float) -> None:
    """
    Place labels around a polar plot at a given radius.
    """
    for angle, label in zip(angles[:-1], labels):
        r = radius

        # left / right alignment
        if 0 <= angle <= np.pi / 2 or 3 * np.pi / 2 <= angle <= 2 * np.pi:
            ha = "left"
        else:
            ha = "right"

        ax.text(
            angle, r, label, fontsize=9, color="black", ha=ha, va="center", wrap=True
        )

    return None


def generate_random_palette(num_colors, seed=12):
    # Generate random colors with varied lightness and saturation
    random.seed(seed)

    colors = []

    for _ in range(num_colors):
        h = random.random()  # Random hue between 0 and 1
        l = random.uniform(0.2, 0.8)  # Lightness between 0.3 and 0.9 for contrast
        s = random.uniform(0.5, 1.0)  # Saturation between 0.6 and 1.0 for vivid colors
        color = colorsys.hls_to_rgb(h, l, s)  # Convert HLS to RGB color
        colors.append(color)

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
    """
    Draw a radar plot on an existing polar matplotlib axis.
    """
    if model_ids == ["ACH-000323", "ACH-002083", "ACH-002228"]:
        model_colors = {
            "ACH-000323": "#00FFFF",  # cyan
            "ACH-002083": "#FF69B4",  # pink
            "ACH-002228": "#800080",  # purple
        }
    else:
        model_colors = generate_model_color_map(model_ids, seed=12)

    # Top 5 per model, union of all
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

    # Angles
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += [angles[0]]

    # Reference circles
    circle_levels = np.linspace(0, global_max, num_circles + 1)[1:]
    for r in circle_levels:
        circle_vals = [r] * (n_vars + 1)
        ax.plot(
            angles, circle_vals, color="grey", linestyle="--", linewidth=1, zorder=0
        )
    # Plot each model
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

    # Axis styling
    ax.set_ylim(0, global_max * 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Labels
    label_radius = global_max * 1.4
    place_labels_polar(ax, angles, labels, label_radius)

    # Title
    ax.set_title(title, fontsize=18, pad=30, fontweight="bold", y=1)

    return None


def make_dropdown_pca_with_selection(df, title="PCA Interactive Plot"):
    out = widgets.Output()  # Create a fresh output for this plot!

    # Prepare PCA input
    pca_input = df.drop(
        columns=["OncotreePrimaryDisease", "source", "ModelID"], errors="ignore"
    ).fillna(0)
    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(pca_input)

    # Add PCA coordinates to the dataframe
    df["PCA1"] = embedding[:, 0]
    df["PCA2"] = embedding[:, 1]

    # Prepare color map for each cancer type
    cancer_types = df["OncotreePrimaryDisease"].unique()
    color_map = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Dark24
    )
    highlight_color_map = {
        cancer: color_map[i % len(color_map)] for i, cancer in enumerate(cancer_types)
    }

    # Create one trace per cancer type
    traces = []
    for cancer in cancer_types:
        df_subset = df[df["OncotreePrimaryDisease"] == cancer]
        trace = go.Scattergl(
            x=df_subset["PCA1"],
            y=df_subset["PCA2"],
            mode="markers",
            name=cancer,
            marker=dict(size=7, color=highlight_color_map[cancer]),
            text=[f"{cancer} | {model_id}" for model_id in df_subset["ModelID"]],
            customdata=df_subset["ModelID"],  # Attach ModelID to customdata
            hoverinfo="text",
        )
        traces.append(trace)

    # Dropdown buttons to highlight cancer types
    dropdown_buttons = []
    for i, cancer in enumerate(cancer_types):
        visibility = [True] * len(cancer_types)
        colors = ["lightgrey"] * len(cancer_types)
        colors[i] = highlight_color_map[cancer]

        button = dict(
            method="update",
            label=cancer,
            args=[
                {
                    "visible": visibility,
                    "marker": [{"color": colors[j]} for j in range(len(cancer_types))],
                },
                {"title": f"PCA Highlighted: {cancer}"},
            ],
        )
        dropdown_buttons.append(button)

    # Default color button
    default_colors = [highlight_color_map[cancer] for cancer in cancer_types]

    fig = go.Figure(data=traces)

    for i, trace in enumerate(fig.data):
        trace.marker.color = default_colors[i]

    fig.update_layout(
        title=title,
        xaxis_title="PCA1",
        yaxis_title="PCA2",
        updatemenus=[
            {
                "direction": "down",
                "showactive": True,
                "x": 1.15,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        dragmode="lasso",  # Ensure Lasso is enabled
        width=1500,
        height=800,
    )

    return fig, out  # Return both the figure and the output widget for selection


def load_model_data(dependency_file, gene_dict_file):
    """
    Load and preprocess gene dependency data and gene dictionary.

    Parameters:
    - dependency_file (str): Path to the gene dependency data file.
    - gene_dict_file (str): Path to the gene dictionary file.

    Returns:
    - dependency_df (DataFrame): Preprocessed gene dependency data.
    - gene_dict_df (DataFrame): Preprocessed gene dictionary.
    """
    # Load gene dependency data
    dependency_df = pd.read_parquet(dependency_file)

    print(dependency_df.shape)
    dependency_df.head(3)

    # Load gene dictionary and filter by QC
    gene_dict_df = (
        pd.read_parquet(gene_dict_file).query("qc_pass").reset_index(drop=True)
    )
    gene_dict_df.entrez_id = gene_dict_df.entrez_id.astype(str)

    # Recode column names to entrez ids
    entrez_genes = [
        x[1].strip(")").strip()
        for x in dependency_df.iloc[:, 1:].columns.str.split("(")
    ]
    entrez_intersection = list(
        set(gene_dict_df.entrez_id).intersection(set(entrez_genes))
    )

    gene_dict_df = gene_dict_df.set_index("entrez_id").reindex(entrez_intersection)

    # Subset dependencies to the genes that passed QC
    dependency_df.columns = ["ModelID"] + entrez_genes
    dependency_df = dependency_df.loc[:, ["ModelID"] + gene_dict_df.index.tolist()]
    dependency_df.columns = ["ModelID"] + gene_dict_df.symbol_id.tolist()
    dependency_df = dependency_df.dropna(axis="columns")

    return dependency_df, gene_dict_df
