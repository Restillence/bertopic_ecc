# file_handling.py
import pandas as pd
import numpy as np

def clean_csv_file(filepath):
    cleaned_file_path = filepath.replace(".csv", "_cleaned.csv")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df.to_csv(cleaned_file_path, index=False)
        return cleaned_file_path
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None

def read_index_file(index_file_path):
    index_file = pd.read_csv(index_file_path, sep=";")
    return index_file

def create_ecc_sample(sample_size, index_file, folderpath_ecc):
    random_start_index = np.random.randint(0, len(index_file)-sample_size)