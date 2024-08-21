import os
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
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
model_save_path = config["model_save_path"]

# Initialize UMAP model
umap_model = UMAP(**config["umap_model_params"])

# Initialize HDBSCAN model
hdbscan_model = HDBSCAN(**config["hdbscan_model_params"])

# Initialize CountVectorizer model
vectorizer_model = CountVectorizer(
    ngram_range=tuple(config["vectorizer_model_params"]["ngram_range"]),
    stop_words=config["vectorizer_model_params"]["stop_words"],
    max_df=config["vectorizer_model_params"]["max_df"],
    min_df=config["vectorizer_model_params"]["min_df"]
)

class BERTopicTrainer:
    def __init__(self, file_handler, text_processor, umap_model, hdbscan_model, vectorizer_model, model_save_path):
        self.file_handler = file_handler
        self.text_processor = text_processor
        self.umap_model = umap_model
        self.hdbscan_model = hdbscan_model
        self.vectorizer_model = vectorizer_model
        self.model_save_path = model_save_path

        # Initialize BERTopic without a representation model, since we're using precomputed embeddings
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            min_topic_size=config["min_topic_size"],
            nr_topics=config["nr_topics"],
            representation_model=None  # Disable the representation model
        )

    def train(self):
        np.random.seed(random_seed)

        # Load precomputed embeddings
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path}")

        # Create the sample and extract relevant sections
        index_file = self.file_handler.read_index_file()
        ecc_sample = self.file_handler.create_ecc_sample(sample_size)
        
        all_relevant_sections = self.text_processor.extract_all_relevant_sections(ecc_sample, max_documents)
        
        # Fit the BERTopic model with precomputed embeddings
        if all_relevant_sections:
            print("Fitting BERTopic...")
            start_time = time.time()
            topics, probabilities = self.topic_model.fit_transform(all_relevant_sections, embeddings=embeddings)
            end_time = time.time()
            print(f"BERTopic model trained on {len(all_relevant_sections)} sections.")
            print(f"Number of topics generated: {len(set(topics))}")
            print(f"Training time: {end_time - start_time:.2f} seconds.")
            self.topic_model.save(self.model_save_path)
            print(f"BERTopic model saved to {self.model_save_path}.")
        else:
            print("No relevant sections found to fit BERTopic.")

if __name__ == "__main__":
    # Initialize FileHandler and TextProcessor
    file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
    text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)
    
    # Initialize and run BERTopicTrainer
    trainer = BERTopicTrainer(file_handler, text_processor, umap_model, hdbscan_model, vectorizer_model, model_save_path)
    trainer.train()
