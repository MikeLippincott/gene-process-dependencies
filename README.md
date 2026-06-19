# Gene Dependency Representations

## Goal

Current cancer treatments tend to be toxic and leave patients with lifelong side-effects.
The future of drug development is based on synthetic lethality, where the combination of two genetic events results in cell death.
It is used in molecular targeted cancer therapy, with the first example of a molecular targeted therapeutic exploiting a synthetic lethal exposed by an inactivated tumor suppressor gene (BRCA1 and 2) receiving FDA approval in 2016 (PARP inhibitor).
The benefits of synthetic lethality-based treatment strategies include success against the majority of cancer mutations, simple identification of treatment-responding patients due to its selective nature of specific cancer cell genetic mutations, and reduced toxicity compared to traditional chemotherapy.

**The goal of this project is to discover multivariate gene vulnerability patterns in cancer.**
Using cancer cell line data from DepMap, we can find multivariate gene vulnerability patterns that can be applied to the development of novel cancer treatments. ​
We apply machine learning to gene knockout data to discover multivariate gene vulnerabilities.
We will apply statistical anaylses to determine the differences in multivariate gene vulnerabilities between pediatric and adult cancers.
Once we discover significant multigene vulnerabilities patterns, we hope to inform drug discovery to develop cancer treatments targeting these vulnerabilities.

## Data

### Access

All data are publicly available.

Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).

## Repository Structure:

This repository is structured as follows:

| Order                                     | Module                                   | Description                                                                                                                                                                                                                                                        |
| :---------------------------------------- | :--------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [0.data-download](0.data-download/)       | Download required files                  | Download gene effect data and cell line information, and download gene QC and construct gene filtering dictionary                                                                                                                                                  |
| [1.data-exploration](1.data-exploration/) | Explore and visualize data               | Create figures to visualize cell line information and split gene effect data into balanced test and train dataframes                                                                                                                                               |
| [2.train-VAE](2.train-VAE/)               | Train Beta VAE and Beta TC VAE models    | Optimize hyperparameters and train Beta Variational Autoencoder/Beta Total Correlation Variational Autoencoder with optimal hyperparameters and previously created test and train dataframes                                                                       |
| [3.analysis](3.analysis/)                 | Analyze Beta VAE and Beta TC VAE Outputs | Generate heatmaps to visualize death windows by cell line and by genes, run Gene Set Enrichment Analysis with BVAE and BTCVAE synthesized data, and analyze extracted BVAE/BTCVAE latent space data to compare similarity of cancer between different demographics |

## Environment Setup

Perform the following steps to set up the `gene_dependency_representations` environment necessary for processing data in this repository.

### Step 1: Create Gene Dependency Representations Environment

```sh
# Run this command to create the proper conda environment (conda version 24.5.0)

conda env create --yes --file environment.yml
```

### Step 2: Activate Gene Dependency Representations Environment

```sh
# Run this command to activate the conda environment for Gene Dependency Representations

conda activate gene_dependency_representations
```

## Webapp

An interactive Streamlit dashboard for exploring gene dependency outputs lives in [`9.webapp/`](9.webapp/).
It is self-contained with its own `uv`-managed environment — no conda setup required.

### Prerequisites

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run locally

```sh
cd 9.webapp
uv run streamlit run app.py
```

`uv` will create a virtual environment and install all dependencies automatically on first run.

### Deploy (headless)

```sh
cd 9.webapp
just deploy
```

This runs Streamlit on `0.0.0.0:8501` by default. Override the port with `PORT=8080 just deploy`.

### Data

The webapp reads pre-computed result files that are committed to the repository:

| File | Description |
| :--- | :---------- |
| `9.webapp/data/Model.parquet` | DepMap cell line metadata |
| `9.webapp/data/pca_embeddings_single_dependencies.parquet` | PCA of raw gene dependency scores |
| `9.webapp/data/pca_embeddings_latent_reactome.parquet` | PCA of Reactome latent scores |
| `9.webapp/data/pca_embeddings_latent_corum.parquet` | PCA of CORUM latent scores |
| `9.webapp/data/pca_embeddings_latent_drug.parquet` | PCA of drug latent scores |
| `5.drug-dependency/results/all_*_results.parquet` | Full latent score tables (scores + spider plots) |

Raw DepMap files (`CRISPRGeneEffect.parquet` etc.) are **not** required by the webapp and are gitignored. Download them separately via [`0.data-download/`](0.data-download/) if you need to re-run the analysis pipeline.
