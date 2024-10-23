import os
import json
import numpy as np
import torch  # For checking if GPU is available
import time  # For time tracking
import threading  # For heartbeat functionality

# Disable parallelism in tokenizers to prevent CPU overutilization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from utils import print_configuration
import datetime

class BertopicModel:
    def __init__(self, config):
        """
        Initialize the BertopicModel class.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration for the model.
        """
        self.config = config
        self.use_gpu = self.config.get("use_gpu", True)  # For UMAP and HDBSCAN
        self.model_save_path = config["model_save_path"]
        self.modeling_type = config.get("modeling_type", "regular")  # Options: ["regular", "zeroshot"]
        self.device = self._select_embedding_device()  # Device for embeddings
        self.topic_model = None
        self.nr_topics = config.get("nr_topics", None)
        self.model = self._select_embedding_model(config)
        self.docs = None  # Initialize self.docs

        # Select UMAP and HDBSCAN implementations
        self._select_umap_hdbscan()

    def _select_embedding_device(self):
        """Select device for embeddings based on GPU availability."""
        if torch.cuda.is_available():
            print("GPU is available. Using GPU for embeddings...")
            return torch.device("cuda")
        else:
            print("GPU not available. Using CPU for embeddings...")
            return torch.device("cpu")

    def _select_umap_hdbscan(self):
        """Select UMAP and HDBSCAN implementations based on config setting and availability."""
        if self.use_gpu and torch.cuda.is_available():
            print("Using GPU-accelerated UMAP and HDBSCAN with cuML...")
            from cuml.manifold import UMAP as cumlUMAP
            from cuml.cluster import HDBSCAN as cumlHDBSCAN
            self.UMAP = cumlUMAP
            self.HDBSCAN = cumlHDBSCAN
            self.use_gpu_umap = True  # Track UMAP implementation
        else:
            print("Using CPU versions of UMAP and HDBSCAN...")
            from umap import UMAP as cpuUMAP
            from hdbscan import HDBSCAN as cpuHDBSCAN
            self.UMAP = cpuUMAP
            self.HDBSCAN = cpuHDBSCAN
            self.use_gpu_umap = False  # Track UMAP implementation

    def _select_embedding_model(self, config):
        """Select the embedding model based on the config setting."""
        embedding_choice = config.get("embedding_model_choice", "all-MiniLM-L12-v2")

        if embedding_choice == "all-MiniLM-L12-v2":
            print("Loading SentenceTransformer model: all-MiniLM-L12-v2...")
            return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=self.device)
        elif embedding_choice == "all-MiniLM-L6-v2":
            print("Loading SentenceTransformer model: all-MiniLM-L6-v2...")
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        else:
            raise ValueError(f"Unknown embedding model choice: {embedding_choice}")

    def _initialize_bertopic_model(self):
        """Initialize the BERTopic model with the specified parameters."""
        print(f"Embedding Model used: {self.model}...")
        num_docs = len(self.docs)
        n_neighbors_config = self.config["umap_model_params"]["n_neighbors"]
        n_neighbors = min(n_neighbors_config, num_docs - 1)
        n_neighbors = max(n_neighbors, 2)  # Ensure n_neighbors is at least 2

        # Adjust n_neighbors for topic embeddings if zero-shot modeling
        if self.modeling_type == "zeroshot":
            num_topics = len(self.config["zeroshot_topic_list"])
            n_neighbors_topics = min(n_neighbors_config, num_topics - 1)
            n_neighbors_topics = max(n_neighbors_topics, 2)
            n_neighbors = min(n_neighbors, n_neighbors_topics)
            print(f"Adjusted n_neighbors to {n_neighbors} based on {num_docs} documents and {num_topics} topics.")
        else:
            print(f"Using n_neighbors {n_neighbors} for dataset size of {num_docs} documents.")

        # Initialize CountVectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=tuple(self.config["vectorizer_model_params"]["ngram_range"]),
            stop_words=self.config["vectorizer_model_params"]["stop_words"],
            min_df=self.config["vectorizer_model_params"]["min_df"]
        )

        # Build UMAP parameters
        umap_params = {
            'n_neighbors': n_neighbors,
            'n_components': self.config["umap_model_params"]["n_components"],
            'min_dist': self.config["umap_model_params"]["min_dist"],
            'metric': self.config["umap_model_params"]["metric"],
            'random_state': 42
        }

        # Add 'low_memory' parameter only if using CPU UMAP
        if not self.use_gpu_umap:
            umap_params['low_memory'] = self.config["umap_model_params"]["low_memory"]

        # Initialize UMAP with adjusted parameters
        umap_model = self.UMAP(**umap_params)

        # Initialize HDBSCAN with specified parameters
        hdbscan_params = {
            'min_cluster_size': self.config["hdbscan_model_params"]["min_cluster_size"],
            'metric': self.config["hdbscan_model_params"]["metric"],
            'cluster_selection_method': self.config["hdbscan_model_params"]["cluster_selection_method"],
            'prediction_data': self.config["hdbscan_model_params"]["prediction_data"]
        }
        hdbscan_model = self.HDBSCAN(**hdbscan_params)

        # Initialize representation models
        keybert_model = KeyBERTInspired(top_n_words=self.config["keybert_params"]["top_n_words"])
        mmr_model = MaximalMarginalRelevance(diversity=self.config["mmr_params"]["diversity"])

        # Initialize BERTopic model
        bertopic_params = {
            'embedding_model': self.model,
            'umap_model': umap_model,
            'hdbscan_model': hdbscan_model,
            'vectorizer_model': vectorizer_model,
            'representation_model': [keybert_model, mmr_model],
            'calculate_probabilities': False,
            'nr_topics': self.nr_topics
        }

        if self.modeling_type == "zeroshot":
            bertopic_params['zeroshot_topic_list'] = self.config.get("zeroshot_topic_list", None)
            bertopic_params['zeroshot_min_similarity'] = self.config.get("zeroshot_min_similarity", None)
            bertopic_params['min_topic_size'] = self.config.get("min_topic_size", 150)

        return BERTopic(**bertopic_params)


    def _print_gpu_usage(self):
        if torch.cuda.is_available():
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU {gpu.id} - Memory Usage: {gpu.memoryUsed}/{gpu.memoryTotal} MB - Utilization: {gpu.load*100}%")

    def train(self, docs):
        """Train the BERTopic model using the specified modeling type.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        None
        """
        self.docs = docs  # Store documents for use in other methods

        # Start embeddings and training time tracking
        embeddings_and_training_start_time = time.time()

        # Train the model using the regular approach
        self._train_regular(docs)

        # End embeddings and training time tracking
        embeddings_and_training_end_time = time.time()
        embeddings_and_training_duration = embeddings_and_training_end_time - embeddings_and_training_start_time
        print(f"Computing embeddings and training took {embeddings_and_training_duration:.2f} seconds.")

        # After training, display topic information and customize labels if zero-shot modeling
        self._post_training_tasks()

    def _train_regular(self, docs):
        # Initialize BERTopic model
        self.topic_model = self._initialize_bertopic_model()

        # Start embeddings time tracking
        embeddings_start_time = time.time()

        # Compute embeddings on GPU or CPU based on device
        print("Computing embeddings...")
        self._print_gpu_usage()
        embeddings = self.model.encode(docs, show_progress_bar=True, batch_size=self.config["batch_size"])
        self._print_gpu_usage()

        # End embeddings time tracking
        embeddings_end_time = time.time()
        embeddings_duration = embeddings_end_time - embeddings_start_time
        print(f"Computing embeddings took {embeddings_duration:.2f} seconds.")

        # Conditionally perform PCA only if not using zero-shot modeling
        if self.modeling_type != "zeroshot":
            # Start PCA dimensionality reduction
            print("Reducing dimensionality of embeddings before UMAP...")
            pca_components = self.config.get("pca_components", 50)
            pca = PCA(n_components=pca_components, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            print(f"Dimensionality reduced to {pca_components} components for less expensive usage of UMAP.")
        else:
            print("Skipping PCA dimensionality reduction for zero-shot topic modeling.")

        # Start training time tracking
        training_start_time = time.time()

        # Fit the BERTopic model with embeddings
        print(f"Fitting BERTopic model using the following modeling type: {self.modeling_type}...")
        try:
            self.topic_model.fit(docs, embeddings)
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return

        # End training time tracking
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        print(f"Training the model took {training_duration:.2f} seconds.")

        # Print information about the training process
        print(f"BERTopic model trained on {len(docs)} documents.")
        print(f"Number of topics generated: {len(self.topic_model.get_topic_info())}")

        # Save the BERTopic model using safetensors
        try:
            print("Saving BERTopic model...")
            self.topic_model.save(
                self.model_save_path,
                serialization="safetensors",
                save_ctfidf=True
            )
            print(f"BERTopic model saved to {self.model_save_path}.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def _post_training_tasks(self):
        """Perform tasks after training, such as displaying topic info and customizing labels."""
        # Display the number of documents assigned to each topic
        print("\nGetting topic information...")
        topic_info = self.topic_model.get_topic_info()
        print(topic_info[['Topic', 'Count']])

        # Customize topic labels if zero-shot modeling
        if self.modeling_type == "zeroshot":
            print("\nCustomizing topic labels with zero-shot topic names...")
            self._customize_topic_labels()

    def _customize_topic_labels(self):
        """Customize topic labels to include zero-shot topic names followed by top words."""
        # Get zero-shot topic list from config
        zeroshot_topic_list = self.config.get("zeroshot_topic_list", [])
        # Get topic info from the model
        topic_info = self.topic_model.get_topic_info()
        
        # Initialize a dictionary to hold the new labels
        topic_labels = {}
        
        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                topic_labels[topic_id] = "Outliers"
            else:
                # Get the assigned topic name from the 'Name' column
                topic_name = row['Name']
                # Check if the topic name matches one of the zero-shot topics
                if topic_name in zeroshot_topic_list:
                    # It's a zero-shot topic
                    label = topic_name
                else:
                    # It's a new topic generated by BERTopic
                    label = f"Topic {topic_id}"
                # Get the top words for the topic
                top_words = ', '.join([word for word, _ in self.topic_model.get_topic(topic_id)])
                # Combine the topic name with the top words
                topic_labels[topic_id] = f"{label}: {top_words}"
        
        # Update the topic labels in the model
        self.topic_model.set_topic_labels(topic_labels)
        
        # Display updated topic labels
        print("Updated topic labels:")
        updated_topic_info = self.topic_model.get_topic_info()
        print(updated_topic_info[['Topic', 'Name']])

    def save_topic_info(self):
        """Save topic information to a CSV file."""
        topic_info = self.topic_model.get_topic_info()
        output_file = os.path.join(self.config.get("output_dir", "."), "topic_info.csv")
        topic_info.to_csv(output_file, index=False)
        print(f"Topic information saved to {output_file}.")

def heartbeat():
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Heartbeat: The current time is {current_time}")
        time.sleep(300)  # Sleep for 5 minutes (300 seconds)

def main():
    """
    Main entry point of the script.

    This function loads the configuration from config.json, sets the random seed, and extracts the necessary variables from the config.
    Then, it initializes the FileHandler and TextProcessor classes with the imported configuration, creates the ECC sample, and extracts the relevant sections.
    Finally, it trains the BERTopic model and saves it to the specified path.

    Returns
    -------
    None
    """
    # Start the heartbeat thread
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
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
    sample_size = config["sample_size"]
    document_split = config["document_split"]
    section_to_analyze = config["section_to_analyze"]
    max_documents = config["max_documents"]

    # Initialize FileHandler and TextProcessor with the imported configuration
    print("Initializing file handler and text processor...")
    file_handler = FileHandler(config=config)
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
        return

    # Instantiate and train the BERTopic model
    bertopic_model = BertopicModel(config)

    # Start training time tracking
    training_start_time = time.time()
    bertopic_model.train(all_relevant_sections)
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"Training the model took {training_duration:.2f} seconds.")

    print("BERTopic model training and saving completed.")

    # Save topic information to CSV
    bertopic_model.save_topic_info()

    # End total execution time tracking
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total execution time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()
