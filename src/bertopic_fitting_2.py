import os
import json
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from bertopic import BERTopic
from file_handling import FileHandler
from text_processing import TextProcessor
from utils import print_configuration
from sentence_transformers import SentenceTransformer

class BertopicFitting:
    """
    Class to fit BERTopic models, save results, and generate visualizations for a given ECC sample.
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
        self.output_dir = "model_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

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
        Fit the BERTopic model, save results, and generate visualizations.

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
        self.topic_model.original_documents_ = all_relevant_sections

        # Save the results to CSV
        self.save_results(all_relevant_sections, topics, ecc_sample)

        # Generate and save information and visualizations
        self.save_basic_info()
        self.save_topics_distribution()
        self.generate_additional_visualizations()

    def save_basic_info(self):
        """
        Save basic information about the model.
        """
        start_time = time.time()
        print("Saving basic information...")
        output_file = os.path.join(self.output_dir, "basic_info.txt")
        with open(output_file, 'w') as f:
            # Get the number of topics
            num_topics = len(self.topic_model.get_topic_info())
            f.write(f"Number of Topics: {num_topics}\n\n")
            
            # Get topic frequency (number of documents per topic)
            topic_info = self.topic_model.get_topic_info()
            f.write("Top 5 Topics by Frequency:\n")
            f.write(topic_info.head(5).to_string())
            f.write("\n\n")
    
            # Get top words for a specific topic (example: topic 0)
            example_topic = 0
            top_words = self.topic_model.get_topic(example_topic)
            f.write(f"Top words for Topic {example_topic}:\n")
            for word, score in top_words:
                f.write(f"  - {word}: {score}\n")
    
        end_time = time.time()
        print(f"Basic information saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def save_topics_distribution(self):
        """
        Save a plot of the topic distribution.
        """
        start_time = time.time()
        print("Saving topic distribution...")
        output_file = os.path.join(self.output_dir, "topic_distribution.png")
        topic_info = self.topic_model.get_topic_info()
        plt.figure()
        plt.bar(topic_info['Topic'], topic_info['Count'])
        plt.xlabel('Topic')
        plt.ylabel('Number of Documents')
        plt.title('Topic Distribution')
        plt.savefig(output_file)
        plt.close()
        end_time = time.time()
        print(f"Topic distribution saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def save_visualization(self, fig, output_file, file_format="png"):
        """
        Save a visualization from BERTopic's built-in visualizations.

        Args:
            fig: The figure to save.
            output_file: The path to save the figure.
            file_format: The format to save the figure ('png' or 'html').
        """
        start_time = time.time()
        print(f"Saving visualization to {output_file}...")
        if file_format == "html":
            fig.write_html(output_file)
        else:
            fig.write_image(output_file)
        end_time = time.time()
        print(f"Visualization saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def generate_additional_visualizations(self):
        """
        Generate and save additional visualizations.
        """
        print("Generating additional visualizations...")
        start_time = time.time()
        
        # Visualize Topics
        print("Visualizing topics...")
        fig = self.topic_model.visualize_topics()
        self.save_visualization(fig, os.path.join(self.output_dir, "topics.html"), file_format="html")
    
        # Visualize Topic Hierarchy
        print("Visualizing topic hierarchy...")
        fig = self.topic_model.visualize_hierarchy()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_hierarchy.html"), file_format="html")
    
        # Visualize Topic Terms (BarChart)
        print("Visualizing topic terms...")
        fig = self.topic_model.visualize_barchart()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_barchart.html"), file_format="html")
    
        # Visualize Topic Similarity (Heatmap)
        print("Visualizing topic similarity...")
        fig = self.topic_model.visualize_heatmap()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_heatmap.html"), file_format="html")
    
        # Visualize Term Score Decline (Term Rank)
        print("Visualizing term rank...")
        fig = self.topic_model.visualize_term_rank()
        self.save_visualization(fig, os.path.join(self.output_dir, "term_rank.html"), file_format="html")
    
        # Visualize Topic Probability Distribution
        if self.topic_model.probabilities_ is not None:
            print("Visualizing topic distribution for the first document...")
            fig = self.topic_model.visualize_distribution(self.topic_model.probabilities_[0])
            self.save_visualization(fig, os.path.join(self.output_dir, "topic_distribution.html"), file_format="html")
    
        end_time = time.time()
        print(f"All visualizations generated and saved in {end_time - start_time:.2f} seconds.")

def main():
    """
    Main entry point of the script.

    This function loads the configuration from `config_hlr.json`, sets the random seed, and extracts the necessary variables from the config.
    Then, it initializes the `FileHandler` and `TextProcessor` classes with the imported configuration, creates the ECC sample, and extracts the relevant sections.
    Finally, it fits the BERTopic model, saves the results, and generates visualizations.
    """
    # Load configuration from config.json
    print("Loading configuration...")
    with open('config.json', 'r') as f:
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
    
    # Instantiate BertopicFitting and process the data
    bertopic_fitting = BertopicFitting(config, model_load_path)
    bertopic_fitting.fit_and_save(all_relevant_sections, ecc_sample)

if __name__ == "__main__":
    main()
