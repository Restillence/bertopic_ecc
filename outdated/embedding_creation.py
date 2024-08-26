import os
import numpy as np
import torch
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from transformers import AutoModel, AutoTokenizer
import json
import time

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract variables from the config
folderpath_ecc = config["folderpath_ecc"]
index_file_path = config["index_file_path"]
embeddings_path = config["embeddings_path"]
sample_size = config["sample_size"]
document_split = config["document_split"]
random_seed = config["random_seed"]
section_to_analyze = config["section_to_analyze"]
max_documents = config["max_documents"]

class FinBERTEmbedding:
    def __init__(self, model_name="yiyanghkust/finbert-tone", batch_size=8):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size

    def load_model(self):
        try:
            start_time = time.time()
            print(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            end_time = time.time()
            print(f"FinBERT model loaded successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            raise

    def get_embeddings(self, texts):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        print(f"Getting embeddings for {len(texts)} paragraphs/sentences.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        all_embeddings = []
        print("Embedding texts using FinBERT model...")
        start_time = time.time()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
            all_embeddings.append(embeddings)

        end_time = time.time()
        print(f"Embeddings created in {end_time - start_time:.2f} seconds.")
        return np.vstack(all_embeddings)

def create_and_save_embeddings():
    total_start_time = time.time()
    np.random.seed(random_seed)

    # Initialize FileHandler and TextProcessor
    file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
    text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

    # Create the sample and extract relevant sections
    index_file = file_handler.read_index_file()
    ecc_sample = file_handler.create_ecc_sample(sample_size)

    all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)

    # Get embeddings from FinBERT
    finbert_embedding = FinBERTEmbedding()
    embeddings = finbert_embedding.get_embeddings(all_relevant_sections)

    # Save the embeddings
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    create_and_save_embeddings()
