---
title: Gene Process Dependency Explorer
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: bsd-3-clause
---

# Gene Process Dependency Explorer

An interactive Streamlit dashboard for exploring gene process dependencies across cancer types, built on [DepMap](https://depmap.org) data and latent representations learned by a Beta Variational Autoencoder (BVAE).

This app accompanies the paper:

> **Discovering multivariate gene vulnerability patterns in cancer using latent representations of gene dependencies**
> WayScience · [GitHub](https://github.com/WayScience/gene_dependency_representations)

---

## What the app shows

The dashboard has six tabs:

| Tab | Description |
| :-- | :---------- |
| **Welcome** | Overview and links |
| **Single dependency plots** | PCA of raw CRISPR gene dependency scores across all cell lines; filter by cancer type |
| **Latent plots** | PCA of VAE latent representations for Reactome, CORUM, and Drug subsets |
| **Spider plots** | Radar charts of biological process dependency profiles per cell line or cancer type |
| **Top scores** | Heatmaps and bar charts of highest-scoring processes by cancer type and model ID |
| **Data table** | Searchable table of all cell lines and their primary diseases |

---

## Running locally

The webapp is self-contained with its own [`uv`](https://docs.astral.sh/uv/)-managed environment.

### Prerequisites

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Start the app

```sh
cd 9.webapp
uv run streamlit run app.py
```

`uv` creates and populates the virtual environment automatically on first run. The app will be available at `http://localhost:8501`.

### Deploy headless (e.g. on a server)

```sh
cd 9.webapp
PORT=8501 uv run streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.port "$PORT" \
  --server.fileWatcherType none
```

---

## Data

All data files are committed to the repository — no external downloads needed to run the app.

| File | Size | Description |
| :--- | ---: | :---------- |
| `data/Model.parquet` | 188 KB | DepMap cell line metadata |
| `data/pca_embeddings_single_dependencies.parquet` | 34 KB | Pre-computed PCA of raw gene dependency scores |
| `data/pca_embeddings_latent_reactome.parquet` | 27 KB | Pre-computed PCA of Reactome latent scores |
| `data/pca_embeddings_latent_corum.parquet` | 27 KB | Pre-computed PCA of CORUM latent scores |
| `data/pca_embeddings_latent_drug.parquet` | 27 KB | Pre-computed PCA of drug latent scores |
| `data/all_reactome_results.parquet` | 1.7 MB | Full Reactome latent scores (spider + heatmap tabs) |
| `data/all_corum_results.parquet` | 1.7 MB | Full CORUM latent scores |
| `data/all_drug_results.parquet` | 4.1 MB | Full drug latent scores |

Raw DepMap files (`CRISPRGeneEffect.parquet` etc.) are **not** required by the webapp and are gitignored. Download them via [`0.data-download/`](../0.data-download/) if you need to re-run the full analysis pipeline.

---

## Repository structure

This webapp lives in `9.webapp/` within the larger [gene_dependency_representations](https://github.com/WayScience/gene_dependency_representations) repository, which contains the full analysis pipeline from data download through VAE training to result generation.
