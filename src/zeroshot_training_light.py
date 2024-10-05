#!/usr/bin/env python3
import os
import json
import numpy as np
import torch  # For checking if GPU is available
import time  # For time tracking
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import print_configuration

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to config_hlr.json relative to the script's directory
config_path = os.path.join(script_dir, '..', 'config_hlr.json')

# Start total execution time tracking
total_start_time = time.time()

# Load configuration from config.json
print("Loading configuration...")
try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    print_configuration(config)
except FileNotFoundError:
    print(f"Configuration file not found at {config_path}. Please ensure the file exists.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from the configuration file: {e}")
    exit(1)

# Set random seed
random_seed = config.get("random_seed", 42)  # Default to 42 if not specified
np.random.seed(random_seed)

# Extract variables from the config
index_file_ecc_folder = config.get("index_file_ecc_folder", "")
folderpath_ecc = config.get("folderpath_ecc", "")
sample_size = config.get("sample_size", 100)
document_split = config.get("document_split", "default_method")
section_to_analyze = config.get("section_to_analyze", "default_section")
max_documents = config.get("max_documents", 1000000)
model_save_path = config.get("model_save_path", "bertopic_model")

# Initialize FileHandler and TextProcessor with the imported configuration
print("Initializing file handler and text processor...")
file_handler = FileHandler(index_file_path=config.get("index_file_path", ""), folderpath_ecc=folderpath_ecc)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

# Start splitting process time tracking
splitting_start_time = time.time()

# Create the sample and extract relevant sections
print("Reading index file and creating ECC sample...")
index_file = file_handler.read_index_file()
ecc_sample = file_handler.create_ecc_sample(sample_size)
all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)

# End splitting process time tracking
splitting_end_time = time.time()
splitting_duration = splitting_end_time - splitting_start_time
print(f"Splitting process took {splitting_duration:.2f} seconds.")

if not all_relevant_sections:
    print("No relevant sections found to fit BERTopic.")
    exit(0)

docs = all_relevant_sections

zeroshot_topic_list = [
      "Welcome to the Conference Call",
      "Revenue and Sales",
      "Expenses and Costs",
      "Earnings and Profit",
      "Marketing",
      "Strategy",
      "Risk and Forward Looking statements"
]

# **GPU Usage Addition Starts Here**

# Determine the device to use (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the embedding model on the selected device
embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)

# **GPU Usage Addition Ends Here**

# Initialize BERTopic with the embedding model
topic_model = BERTopic(
    embedding_model=embedding_model,  # Use the loaded embedding model
    min_topic_size=100,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=0.0,
    representation_model=KeyBERTInspired()  # Removed 'model=embedding_model'
)

# Start training time tracking
print("Training BERTopic model...")
training_start_time = time.time()
topics, _ = topic_model.fit_transform(docs)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Training process took {training_duration:.2f} seconds.")

# Save the BERTopic model using safetensors
try:
    print("Saving BERTopic model...")
    topic_model.save(
        model_save_path,
        serialization="safetensors",
        save_ctfidf=True
    )
    print(f"BERTopic model saved to {model_save_path}.")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

print("BERTopic model training and saving completed.")

# End total execution time tracking
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"Total execution time: {total_duration:.2f} seconds.")

# We fit our model using the zero-shot topics
# and we define a minimum similarity. For each document,
# if the similarity does not exceed that value, it will be used
# for clustering instead.
