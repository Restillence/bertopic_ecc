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

# Start total execution time tracking
total_start_time = time.time()

# Load configuration from config.json
print("Loading configuration...")
with open('config_hlr.json', 'r') as config_file:
    config = json.load(config_file)
print_configuration(config)

# Set random seed
random_seed = config["random_seed"]
np.random.seed(random_seed)

# Extract variables from the config
index_file_ecc_folder = config["index_file_ecc_folder"]
folderpath_ecc = config["folderpath_ecc"]
sample_size = config["sample_size"]
document_split = config["document_split"]
section_to_analyze = config["section_to_analyze"]
max_documents = config["max_documents"]
model_save_path = config["model_save_path"]

# Initialize FileHandler and TextProcessor with the imported configuration
print("Initializing file handler and text processor...")
file_handler = FileHandler(index_file_path=config["index_file_path"], folderpath_ecc=folderpath_ecc)
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
    min_topic_size=50,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=0.1,
    representation_model=KeyBERTInspired(model=embedding_model)  # Ensure representation model uses the same embedding model
)

# Start training time tracking
print("Training BERTopic model...")
training_start_time = time.time()
topics, _ = topic_model.fit_transform(docs)
training_end_time = time.time()
training_duration = training_end_time - training_start_time

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
