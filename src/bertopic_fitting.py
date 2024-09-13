import os
import json
import time
import pandas as pd
import numpy as np
from bertopic import BERTopic
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from utils import print_configuration, load_bertopic_model
from sentence_transformers import SentenceTransformer
import torch

class BertopicFitting:
    """
    Class to fit and save BERTopic models for a given ECC sample.

    Attributes:
        config (dict): Configuration dictionary containing parameters for the BERTopic model.
        model_load_path (str): Filepath to the pre-trained BERTopic model.
        topic_model (BERTopic): Pre-trained BERTopic model loaded from the given filepath.
        index_file_ecc_folder (str): Folderpath to the ECC index file.
    """

    def __init__(self, config, model_load_path):
        """
        Initialize the BertopicFitting class.

        Args:
            config (dict): Configuration dictionary containing parameters for the BERTopic model.
            model_load_path (str): Filepath to the pre-trained BERTopic model.
        """
        self.config = config
        self.model_load_path = model_load_path
        self.topic_model = self._load_bertopic_model()
        self.index_file_ecc_folder = config["index_file_ecc_folder"]

    def _load_bertopic_model(self):
        """
        Load the pre-trained BERTopic model from the given filepath and use GPU for the embedding model if available,
        otherwise fall back to CPU.
        """
        print(f"Loading BERTopic model from {self.model_load_path}...")
        
        # Check if a GPU is available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the embedding model on the selected device
        embedding_model = SentenceTransformer(self.config["embedding_model_choice"], device=device)
        print(f"Embedding model loaded successfully. On device: {device}")
        
        # Load the BERTopic model with the embedding model
        return BERTopic.load(self.model_load_path, embedding_model=embedding_model)

    def save_results(self, all_relevant_sections, topics, ecc_sample):
        """
        Save the results of the BERTopic model to a CSV file.

        Args:
            all_relevant_sections (list): List of all relevant sections from the ECC sample.
            topics (numpy.ndarray): Array of topics assigned to each section.
            ecc_sample (dict): Dictionary containing the ECC sample data.
        """
        result_dict = {}
        topic_idx = 0

        for permco, calls in ecc_sample.items():
            for call_id, value in calls.items():
                sections = value[-1]  # Assuming the last element in value is the list of sections
                num_sections = len(sections)
                section_topics = topics[topic_idx:topic_idx + num_sections]

                # Ensure topics and text lists have the same length
                if len(section_topics) != num_sections:
                    raise ValueError(f"Mismatch between number of topics and sections for call ID: {call_id}")

                # Convert the section topics from NumPy array to a list
                section_topics = section_topics.tolist()

                # Format sections and topics to be stored correctly in the CSV
                result_dict[call_id] = {
                    "permco": permco,
                    "company_info": value[0],
                    "date": value[1],
                    "sections": sections,  # Keeping it as a list for CSV storage
                    "topics": section_topics  # Convert lists to JSON strings
                }
                topic_idx += num_sections

        # Convert results to DataFrame
        records = []
        for call_id, call_data in result_dict.items():
            records.append({
                'permco': call_data['permco'],
                'call_id': call_id,
                'company_info': call_data['company_info'],
                'date': call_data['date'],
                'text': json.dumps(call_data['sections']),  # Convert lists to JSON strings
                'topics': json.dumps(call_data['topics'])   # Convert lists to JSON strings
            })

        results_df = pd.DataFrame(records)
        results_output_path = os.path.join(self.index_file_ecc_folder, 'topics_output.csv')
        results_df.to_csv(results_output_path, index=False)
        print(f"Results saved to {results_output_path}.")

    def fit_and_save(self, all_relevant_sections, ecc_sample):
        """
        Fit the BERTopic model and save the results to a CSV file.
        Save the BERTopic model with the transformed documents and original untransformed documents.
        
        Args:
            all_relevant_sections (list): List of all relevant sections from the ECC sample.
            ecc_sample (dict): Dictionary containing the ECC sample data.
        """
        bertopic_start_time = time.time()
        print("Transforming documents with the BERTopic model...")
        
        # Transform documents
        topics, probabilities = self.topic_model.transform(all_relevant_sections)
        
        end_time = time.time()
        print(f"BERTopic model transformed {len(all_relevant_sections)} sections.")
        print(f"Transformation time: {end_time - bertopic_start_time:.2f} seconds.")
        
        # Save the topics and probabilities to the BERTopic model
        self.topic_model.topics_ = topics
        self.topic_model.probabilities_ = probabilities
        
        # Save both the original (untransformed) documents and the transformed documents
        self.topic_model.original_documents_ = all_relevant_sections

        # Ensure the model is saved in a CPU-friendly format
        print("Moving model to CPU for saving...")
        device = torch.device('cpu')
        self.topic_model.embedding_model.to(device)  # Move the embedding model to CPU

        # Save the model with transformed data and original documents
        model_save_path_with_data = os.path.join(self.model_load_path, 'bertopic_model_with_data_and_docs_cpu')
        self.topic_model.save(model_save_path_with_data)
        print(f"Model with transformed data and original documents saved to {model_save_path_with_data}.")
        
        # Save the results to CSV as before
        self.save_results(all_relevant_sections, topics, ecc_sample)

def main():
    """
    Main entry point of the script.

    This function loads the configuration from `config.json`, sets the random seed, and extracts the necessary variables from the config.
    Then, it initializes the `FileHandler` and `TextProcessor` classes with the imported configuration, creates the ECC sample, and extracts the relevant sections.
    Finally, it fits the BERTopic model and saves the results to a CSV file, where each section can be mapped to its assigned topic.

    Returns
    -------
    None
    """
    # Load configuration from config.json
    print("Loading configuration...")
    with open('config_hlr.json', 'r') as f:
        config = json.load(f)
    print_configuration(config)

    # Extract variables from the config
    random_seed = config["random_seed"]
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    index_file_ecc_folder = config["index_file_ecc_folder"]
    folderpath_ecc = config["folderpath_ecc"]
    sample_size = config["sample_size"]
    document_split = config["document_split"]
    section_to_analyze = config["section_to_analyze"]
    max_documents = config["max_documents"]
    model_load_path = config["model_load_path"]

    # Initialize FileHandler and TextProcessor with the imported configuration
    print("Initializing file handler and text processor...")
    file_handler = FileHandler(index_file_path=config["index_file_path"], folderpath_ecc=folderpath_ecc)
    text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

    # Create the sample and extract relevant sections
    print("Reading index file and creating ECC sample...")
    index_file = file_handler.read_index_file()
    ecc_sample = file_handler.create_ecc_sample(sample_size)

    # Extract texts for BERTopic analysis (processed sections/paragraphs)
    all_relevant_sections = []
    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            relevant_sections = text_processor.extract_and_split_section(permco, call_id, value[0], value[1], value[2])
            all_relevant_sections.extend(relevant_sections)
            # Add the relevant sections to the ECC sample
            ecc_sample[permco][call_id] = (*value, relevant_sections)

    if not all_relevant_sections:
        print("No relevant sections found to fit BERTopic.")
        return

    # Instantiate BertopicFitting and save results
    bertopic_fitting = BertopicFitting(config, model_load_path)
    bertopic_fitting.fit_and_save(all_relevant_sections, ecc_sample)

if __name__ == "__main__":
    main()
