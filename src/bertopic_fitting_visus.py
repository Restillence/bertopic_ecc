# Import necessary libraries
import os
import sys
import json
import time
import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Adjust the path to include 'src' if it's not already in the system path
current_dir = os.getcwd()
if "src" not in current_dir:
    src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
    sys.path.append(src_path)

# Import custom modules
from file_handling import FileHandler
from text_processing import TextProcessor
from utils import print_configuration

# Step 1: Set a random seed for reproducibility
random_seed = 42  # You can change this value as needed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
print(f"Random seed set to {random_seed}.")

# Step 2: Initialize FileHandler and TextProcessor, read index file, create ECC sample, and extract relevant sections
# Load configuration from config.json
config_file_path = r'C:\Users\nikla\OneDrive\Dokumente\winfoMaster\Masterarbeit\bertopic_ecc\config.json'  # Update this path as needed
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Display the loaded configuration
print_configuration(config)

# Extract necessary variables from configuration
index_file_ecc_folder = config.get("index_file_ecc_folder")
folderpath_ecc = config.get("folderpath_ecc")
sample_size = config.get("sample_size")
document_split = config.get("document_split")
section_to_analyze = config.get("section_to_analyze")
max_documents = config.get("max_documents")
model_load_path = config.get("model_load_path")
embedding_model_choice = config.get("embedding_model_choice")
output_dir = config.get("output_dir", "transformation_results")

# Initialize FileHandler and TextProcessor
print("Initializing FileHandler and TextProcessor...")
file_handler = FileHandler(config=config)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

# Read the index file
print("Reading index file...")
index_file = file_handler.read_index_file()

# Create ECC sample
print("Creating ECC sample...")
ecc_sample = file_handler.create_ecc_sample(sample_size)

# Extract texts for BERTopic analysis (processed sections/paragraphs)
print("Extracting and processing relevant sections...")
all_relevant_sections = []
extraction_start_time = time.time()
for permco, calls in ecc_sample.items():
    for call_id, value in calls.items():
        company_info = value.get('company_name', '')
        date = value.get('date', '')
        text = value.get('text_content', '')
        relevant_sections = text_processor.extract_and_split_section(permco, call_id, company_info, date, text)
        all_relevant_sections.extend(relevant_sections)
        # Add the relevant sections to the ECC sample
        value['relevant_sections'] = relevant_sections
extraction_end_time = time.time()
print(f"Extraction and processing completed in {extraction_end_time - extraction_start_time:.2f} seconds.")

if not all_relevant_sections:
    print("No relevant sections found to transform with BERTopic.")
else:
    # Step 3: Load the embedding model and BERTopic model, compute embeddings, and transform documents
    # Determine the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the SentenceTransformer embedding model
    print(f"Loading SentenceTransformer model: {embedding_model_choice}...")
    embedding_model = SentenceTransformer(embedding_model_choice, device=device)
    print("SentenceTransformer embedding model loaded successfully.")

    # Load the pre-trained BERTopic model
    print(f"Loading BERTopic model from {model_load_path}...")
    topic_model = BERTopic.load(model_load_path, embedding_model=embedding_model)
    print("BERTopic model loaded successfully.")

    # Compute embeddings for the new documents
    print("Computing embeddings for new documents...")
    embeddings_start_time = time.time()
    embeddings = embedding_model.encode(all_relevant_sections, show_progress_bar=True)
    embeddings_end_time = time.time()
    print(f"Computed embeddings for {len(all_relevant_sections)} documents in {embeddings_end_time - embeddings_start_time:.2f} seconds.")

    # Transform documents to get topic assignments and probabilities
    print("Transforming documents with the BERTopic model...")
    transform_start_time = time.time()
    topics, probabilities = topic_model.transform(all_relevant_sections, embeddings)
    transform_end_time = time.time()
    print(f"Transformed documents in {transform_end_time - transform_start_time:.2f} seconds.")

    # (Optional) Save the results to a CSV file
    import pandas as pd
    import json

    print("Saving transformation results to CSV...")
    records = []
    for idx, (topic, prob) in enumerate(zip(topics, probabilities)):
        records.append({
            'Document_ID': idx + 1,
            'Topic': topic,
            'Probability': prob.max(),
            'Probabilities': json.dumps(prob.tolist())
        })
    results_df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    results_output_path = os.path.join(output_dir, 'transformed_topics.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}.")
