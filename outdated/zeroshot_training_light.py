#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch  # For checking if GPU is available
import time  # For time tracking
import threading  # For heartbeat functionality
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler  # Ensure this is correctly implemented
from text_processing import TextProcessor  # Ensure this is correctly implemented
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import print_configuration
from umap import UMAP

def heartbeat():
    """
    Prints a heartbeat message to the console every 5 minutes.
    Runs indefinitely until the main program exits.
    """
    while True:
        time.sleep(300)  # 300 seconds = 5 minutes
        print("[Heartbeat] The script is still running...")

def main():
    # Suppress Hugging Face tokenizers parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start the heartbeat thread
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    # Determine the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to config_hlr.json relative to the script's directory
    config_path = os.path.join(script_dir, '..', 'config.json')

    # Start total execution time tracking
    total_start_time = time.time()

    # Load configuration from config_hlr.json
    print("Loading configuration...")
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        print_configuration(config)
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}. Please ensure the file exists.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the configuration file: {e}")
        sys.exit(1)

    # Set random seed
    random_seed = config.get("random_seed", 42)  # Default to 42 if not specified
    np.random.seed(random_seed)

    # Extract variables from the config
    index_file_ecc_folder = config.get("index_file_ecc_folder", "")
    folderpath_ecc = config.get("folderpath_ecc", "")
    sample_size = config.get("sample_size", 100)
    document_split = config.get("document_split", "default_method")
    section_to_analyze = config.get("section_to_analyze", "default_section")
    max_documents = config.get("max_documents", 1000)
    model_save_path = config.get("model_save_path", "bertopic_model")
    embeddings_save_path = config.get("embeddings_save_path", "model_outputs_zeroshot/embeddings.npy")  # Optional

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
        sys.exit(0)

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

    # **Embedding Computation Starts Here**

    # Determine the device to use (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the embedding model on the selected device
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    print(f"Loaded embedding model: {embedding_model}")

    # Compute embeddings with batch size parameter
    print("Computing embeddings for documents...")
    start_time = time.time()
    embeddings = embedding_model.encode(
        docs,
        batch_size=32,              # Adjust the batch size as needed
        show_progress_bar=True,
        convert_to_numpy=True,      # Ensure embeddings are NumPy arrays
        device=device               # Ensure embeddings are computed on the correct device
    )
    end_time = time.time()
    compute_duration = end_time - start_time
    print(f"Computed embeddings in {compute_duration:.2f} seconds.")

    # Optional: Save embeddings to disk for future use
    # Uncomment the following lines if you wish to save embeddings
    # embeddings_save_full_path = os.path.join(script_dir, '..', embeddings_save_path)
    # os.makedirs(os.path.dirname(embeddings_save_full_path), exist_ok=True)
    # print(f"Saving embeddings to {embeddings_save_full_path}...")
    # np.save(embeddings_save_full_path, embeddings)
    # print("Embeddings saved successfully.")

    # **Embedding Computation Ends Here**

    # Initialize UMAP for dimensionality reduction (optional customization)
    umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=True, random_state=42)

    # Initialize BERTopic with the embedding model
    print("Initializing BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Provide the actual embedding model
        umap_model=umap_model,
        min_topic_size=150,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.05,
        representation_model=KeyBERTInspired()  # No need to pass the model here
    )

    # Start training time tracking
    print("Training BERTopic model...")
    training_start_time = time.time()
    topic_model = topic_model.fit(docs, embeddings=embeddings)  # Pass precomputed embeddings
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"Training process took {training_duration:.2f} seconds.")

    # Save the BERTopic model using safetensors
    try:
        print("Saving BERTopic model...")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
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

if __name__ == "__main__":
    main()
