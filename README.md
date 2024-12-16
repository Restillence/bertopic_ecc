# BERTopic_ECC

Master Thesis Project: Analysis of Earnings Call Transcripts (ECC) using BERTopic

## Overview

This project utilizes BERTopic to analyze Earnings Call Transcripts (ECC) as part of a Master’s thesis. It includes all necessary code to replicate the results, located in the `src` folder.

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Replication Steps](#replication-steps)
- [Usage](#usage)
- [Requirements](#requirements)
- [Notes](#notes)
- [Additional Resources](#additional-resources)

## Setup

Follow these steps to set up the project environment:

1. **Clone the Repository** *(Optional if using GitHub)*

    ```bash
    git clone https://github.com/yourusername/bertopic_ecc.git
    ```

2. **Open in IDE**

    Open the cloned repository in your preferred Integrated Development Environment (IDE).

3. **Install Python & Create Virtual Environment**

    Ensure you have Python version 3.9 or higher installed. It is recommended to create a virtual environment to manage dependencies:

    ```bash
    conda create --name bertopic_env python=3.9
    conda activate bertopic_env
    ```

4. **Install Dependencies**

    Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    If you encounter errors during installation, use the provided installation script that skips problematic lines:

    ```bash
    python src/package_installations.py
    ```

5. **Configure File Paths**

    Update the `config.json` file with your local file paths:

    ```json
    {
      "index_file_ecc_folder": "D:/daten_masterarbeit/",
      "folderpath_ecc": "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/",
      "model_save_path": "D:/daten_masterarbeit/bertopic_model_dir_regular_5000",
      "model_load_path": "D:/daten_masterarbeit/bertopic_model_dir_zeroshot_10000",
      "index_file_path": "D:/daten_masterarbeit/list_earnings_call_transcripts.csv"
    }
    ```

    If the `config.json` file is not found, manually set its filepath in the relevant scripts by searching for `fallback_config_path`.

## Replication Steps

To replicate the results of the study, execute the following scripts in order:

1. **Train BERTopic Model**

    Trains and saves the BERTopic model.

    ```bash
    python src/bertopic_training.py
    ```

2. **Fit Model and Generate Visualizations**

    Fits the trained model on the data and generates visualizations and topic quality evaluation files.

    ```bash
    python src/bertopic_fitting_visus.py
    ```

    **Outputs:**
    - Visualizations
    - Topic Quality Evaluation files
    - CSV containing topic vectors

3. **Prepare Final Dataset**

    Prepares the final dataset, including all dependent, independent, and control variables.

    ```bash
    python src/create_final_dataset.py
    ```

    **Output:**
    - Final dataset ready for analysis

4. **Perform Variable Analysis**

    Conducts regression analysis and generates average transition matrices.

    ```bash
    python src/variable_analysis.py
    ```

    **Outputs:**
    - Regression results
    - Average transition matrices

5. **Optional: Generate Plots and Statistics**

    Generates additional plots and statistics from the dataset.

    ```bash
    python src/run_ecc_analysis.py
    ```

    Note: Some plots are computationally intensive. You may need to comment out unnecessary sections in the main function before running the script. Plots are saved in the `plots` folder.

6. **Explore Data and Models**

    Use the Jupyter notebooks in the `exploration` folder to explore data and BERTopic models further.

## Usage

### Configuration Options

- **Sampling Mode:**
  - `"full_random"`: Trains BERTopic on a lower sample size.
  - `"random_company"`: Fits the model on all companies.

- **Topics to Keep:**
  - `"auto"`: Removes topics not appearing in a specified percentage (e.g., 90%) of companies in at least one call.
  - `"all"` or a specific list of topics.

- **Topic Threshold Percentage:**
  - Configurable in `config.json` (e.g., `90`).

### Recommended Workflow

1. **Train with Full Random Sampling (Sample Size: 10,000):**

    ```bash
    python src/bertopic_training.py --sampling_mode full_random
    ```

2. **Fit with Random Company Sampling (Sample Size: 1,729):**

    ```bash
    python src/bertopic_fitting_visus.py --sampling_mode random_company
    ```

3. **Ensure Filepaths are Adjusted:**

    Verify that the file paths in `create_final_dataset.py` and `variable_analysis.py` match your local setup.

## Requirements

- **Python:** Version 3.9 or higher

- **Python Packages:**
  - pandas
  - numpy
  - os
  - json
  - time
  - bertopic
  - KeyBERT
  - nltk
  - re
  - matplotlib
  - plotly
  - seaborn
  - scikit-learn
  - wordcloud
  - openpyxl
  - textblob
  - networkx
  - transformers
  - torch
  - sentence_transformers
  - GPUtil
  - spacy
  - tqdm

- **Spacy Model for Named Entity Recognition (NER):**

    ```bash
    python -m spacy download en_core_web_sm
    ```

- **cuML Installations for GPU-based HDBSCAN and UMAP:**

    ```bash
    pip install bertopic
    pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
    pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
    pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com
    pip install --upgrade cupy-cuda11x -f https://pip.cupy.dev/aarch64
    ```

## Notes

- **IDE Compatibility:**
  - Developed using Visual Studio Code. Some scripts might require adjustments to work with other IDEs.

- **Data Splitting:**
  - Methods have been manually verified on sample data to ensure accuracy.

- **BERTopic Functionality:**
  - Assigns a topic to every sentence in the transcripts.

- **Configuration Modes:**
  - Supports `"regular"` and `"zeroshot"` modes.
  - Ensure fallback paths in `bertopic_training.py` and `bertopic_fitting_visus.py` are correctly set if `config.json` is missing.

## Additional Resources

### Useful Videos to Get Started with BERTopic

- [Introduction to BERTopic](https://www.youtube.com/watch?v=uZxQz87lb84)
- [Advanced BERTopic Usage](https://www.youtube.com/watch?v=5a5Dfft-rWc)
