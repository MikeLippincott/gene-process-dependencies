#!/usr/bin/env python
# coding: utf-8

# ## Download Dependency Data
#
# Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).
#
# - `CRISPRGeneDependency.parquet`: The data in this document describes the probability that a gene knockdown has an effect on cell-inhibition or death. These probabilities are derived from the data contained in CRISPRGeneEffect.parquet using methods described [here](https://doi.org/10.1101/720243)
# - `Model.parquet`: Metadata for all of DepMap’s cancer models/cell lines.
# - `CRISPRGeneEffect.parquet`: The data in this document are the Gene Effect Scores obtained from CRISPR knockout screens conducted by the Broad Institute. Negative scores notate that cell growth inhibition and/or death occurred following a gene knockout. Information on how these Gene Effect Scores were determined can be found [here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02540-7)
# - `depmap_gene_meta.tsv`: Genes that passed QC and were included in the training model for Pan et al. 2022. We use this data to filter genes as input to our models. The genes were filtered based 1) variance, 2) perturbation confidence, and 3) high on target predictions based on high correlation across other guides.
#
# >Tsherniak A, Vazquez F, Montgomery PG, Weir BA, Kryukov G, Cowley GS, Gill S, Harrington WF, Pantel S, Krill-Burger JM, Meyers RM, Ali L, Goodale A, Lee Y, Jiang G, Hsiao J, Gerath WFJ, Howell S, Merkel E, Ghandi M, Garraway LA, Root DE, Golub TR, Boehm JS, Hahn WC. Defining a Cancer Dependency Map. Cell. 2017 Jul 27;170(3):564-576.

# In[1]:


import pathlib
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa

# In[2]:


def download_dependency_data(figshare_id, figshare_url, output_file):
    """
    Download the provided figshare resource
    """
    urllib.request.urlretrieve(f"{figshare_url}/{figshare_id}", output_file)


# In[3]:


# Set download constants
output_dir = pathlib.Path("data")
figshare_url = "https://ndownloader.figshare.com/files/"

download_dict = {
    "46489732": "Model.csv",
    "46489063": "CRISPRGeneEffect_Uncorrected.csv",
    "46489021": "CRISPRGeneDependency.csv",
    "29094531": "depmap_gene_meta.tsv",
    # DepMap, Broad (2024). DepMap 24Q2 Public. Figshare+. Dataset. https://doi.org/10.25452/figshare.plus.25880521.v1
}


# In[4]:


# Make sure directory exists
output_dir.mkdir(exist_ok=True)


# In[5]:


for figshare_id in download_dict:
    # Set output file
    output_file = pathlib.Path(output_dir, download_dict[figshare_id])

    # Download the dependency data
    print(f"Downloading {output_file}...")

    download_dependency_data(
        figshare_id=figshare_id, figshare_url=figshare_url, output_file=output_file
    )


# In[6]:


# Column name correction for CRISPRGeneEffect
df = pd.read_csv("../0.data-download/data/CRISPRGeneEffect_Uncorrected.csv")
df = df.rename(columns={df.columns[0]: "ModelID"})
filepath = Path("../0.data-download/data/CRISPRGeneEffect.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(filepath, index=False)


# In[7]:


# Convert to parquet
# List of CSV files

data_directory = "../0.data-download/data/"
model_file = pathlib.Path(data_directory, "Model.csv").resolve()
effect_data_file = pathlib.Path(data_directory, "CRISPRGeneEffect.csv").resolve()
dependency_file = pathlib.Path(data_directory, "CRISPRGeneDependency.csv").resolve()
metadata_file = pathlib.Path(data_directory, "depmap_gene_meta.tsv").resolve()

csv_files = [model_file, effect_data_file, dependency_file]

# Convert each CSV to Parquet
for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Define the output Parquet file name
    parquet_file = csv_file.with_suffix(".parquet")

    # Save the DataFrame as a Parquet file
    df.to_parquet(parquet_file, index=False)

meta_df = pd.read_csv(metadata_file, sep="\t")
meta_parquet = metadata_file.with_suffix(".parquet")
meta_df.to_parquet(meta_parquet, index=False)
