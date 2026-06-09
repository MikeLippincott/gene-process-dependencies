#!/bin/bash

# shellcheck disable=SC2035
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ *.ipynb

# check if the data are present
# if not then run the download scripts
# if data are present then skip the download scripts
if [ ! -f "data/Model.parquet" ]; then
    echo "Data files not found. Running data download scripts..."

    uv run python scripts/1.data_downloader.py
    uv run python scripts/2.construct_gene_dictionary.py

    echo "Data download and processing complete."
else
    echo "Data files already exist. Skipping data download."
fi


cp data/Model.parquet ../9.webapp/data/Model.parquet
cp data/CRISPRGeneEffect.parquet ../9.webapp/data/CRISPRGeneEffect.parquet
cp data/CRISPR_gene_dictionary.parquet ../9.webapp/data/CRISPR_gene_dictionary.parquet
