import sys
import os

# Add the oop_code_test directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
import ast  # to safely evaluate string representations of lists
from file_handling import FileHandler
from text_processing import TextProcessor
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract variables from the config
folderpath_ecc = config["folderpath_ecc"]
index_file_ecc_folder = config["index_file_ecc_folder"]
sample_size = config["sample_size"]
document_split = config["document_split"]
random_seed = config["random_seed"]
section_to_analyze = config["section_to_analyze"]
max_documents = config["max_documents"]
index_file_path = config["index_file_path"]

# Initialize FileHandler and TextProcessor
file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

np.random.seed(random_seed)
index_file = file_handler.read_index_file()
ecc_sample = file_handler.create_ecc_sample(sample_size)
all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)

try:
    with open("testfile_splitting_functionality.txt", "w", encoding="utf-8") as f:
        for section in all_relevant_sections:
            f.write(section + "\n\n")
    print("File created successfully!")
except Exception as e:
    print(f"An error occurred: {e}")