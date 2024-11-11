# bertopic_training_2.py
import os
import json
import numpy as np
import torch  # For checking if GPU is available
import time  # For time tracking
import threading  # For heartbeat functionality
import datetime

# Disable parallelism in tokenizers to prevent CPU overutilization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import print_configuration
from scipy.cluster import hierarchy as sch  # For hierarchical topic modeling
from scipy.cluster.hierarchy import fcluster, linkage
from collections import defaultdict



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
        self.embeddings = None  # Initialize embeddings

        # Read the apply_topic_merging parameter from the config
        self.apply_topic_merging = config.get("apply_topic_merging", False)

        # Read the similarity_threshold from the config
        self.similarity_threshold = config.get('similarity_threshold', 0.5)

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
            print(f"Adjusted n_neighbors to {n_neighbors} based on {num_docs} documents and {num_topics}.")
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

        # Initialize ClassTfidfTransformer with all seed words (flattened list)
        seed_words_dict = self.config.get("seed_words", {})
        seed_words = [word for words in seed_words_dict.values() for word in words]

        ctfidf_model = ClassTfidfTransformer(
            seed_words=seed_words,
            seed_multiplier=self.config.get("ctfidf_seed_multiplier", 2)
        )

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
            'ctfidf_model': ctfidf_model,
            'calculate_probabilities': False,
            'nr_topics': self.nr_topics,
            'min_topic_size': self.config.get("min_topic_size", 150)
        }

        if self.modeling_type == "zeroshot":
            bertopic_params['zeroshot_topic_list'] = self.config.get("zeroshot_topic_list", [])
            bertopic_params['zeroshot_min_similarity'] = self.config.get("zeroshot_min_similarity", 0.05)

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

        # After training, perform post-training tasks
        self._post_training_tasks()

        # Save the BERTopic model using safetensors after merging and label updates
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

        # Store embeddings in the class
        self.embeddings = embeddings

        # End embeddings time tracking
        embeddings_end_time = time.time()
        embeddings_duration = embeddings_end_time - embeddings_start_time
        print(f"Computing embeddings took {embeddings_duration:.2f} seconds.")

        # Use embeddings directly without PCA
        print("Using embeddings directly without PCA.")

        # Start training time tracking
        training_start_time = time.time()

        # Fit and transform the BERTopic model with embeddings
        print(f"Fitting and transforming BERTopic model using the following modeling type: {self.modeling_type}...")
        try:
            topics, probs = self.topic_model.fit_transform(docs, embeddings)
            self.topics = topics  # Store topics for later use
            self.probs = probs     # Store probabilities for later use
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

    def _post_training_tasks(self):
        """Perform tasks after training, such as merging topics, customizing labels, and generating hierarchical topics."""
        # Check if topic merging is enabled in the configuration
        if self.apply_topic_merging:
            # Proceed to merge similar topics based on the topic_list
            topic_list = self.config.get("zeroshot_topic_list", [])
            print("\nMerging similar topics based on the provided topic list...")
            self.merge_similar_topics(topic_list)
        else:
            print("\nSkipping topic merging as per configuration.")

        # Customize topic labels if zero-shot modeling
        if self.modeling_type == "zeroshot":
            print("\nCustomizing topic labels with zero-shot topic names...")
            self._customize_topic_labels()

        # After merging and customizing labels, get the topic information
        print("\nGetting updated topic information...")
        topic_info = self.topic_model.get_topic_info()

        # Prepare the topic information dataframe with required columns
        # Add an index column (the default index)
        topic_info.reset_index(inplace=True)
        # Get topic representations
        representations = []
        for topic in topic_info['Topic']:
            if topic == -1:
                representations.append(None)
            else:
                words_weights = self.topic_model.get_topic(topic)
                if words_weights:
                    words = [word for word, _ in words_weights]
                    representations.append(', '.join(words))
                else:
                    representations.append(None)
        topic_info['Representation'] = representations

        # Reorder or select columns as per requirement
        topic_info = topic_info[['index', 'Topic', 'Count', 'Name', 'Representation']]

        # Display the final topic information
        print("\nFinal topic information:")
        print(topic_info)

        # Generate hierarchical topics
        print("\nGenerating hierarchical topics...")
        try:
            hierarchical_topics = self.topic_model.hierarchical_topics(
                self.docs,
                linkage_function=lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
            )
            hierarchical_topics_output_path = os.path.join(self.config.get("output_dir", "."), 'hierarchical_topics.csv')
            hierarchical_topics.to_csv(hierarchical_topics_output_path, index=False)
            print(f"Hierarchical topics saved to {hierarchical_topics_output_path}.")
        except Exception as e:
            print(f"An error occurred while generating hierarchical topics: {e}")
            import traceback
            traceback.print_exc()

        # Save visualize_hierarchy before merging
        print("\nVisualizing hierarchy before merging...")
        try:
            hierarchy_fig_before = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            hierarchy_fig_before.write_html(os.path.join(self.config.get("output_dir", "."), 'topic_hierarchy_before_merging.html'))
            print("Hierarchy visualization before merging saved.")
        except Exception as e:
            print(f"An error occurred while visualizing hierarchy before merging: {e}")
            import traceback
            traceback.print_exc()

        # Now, perform topic merging based on a threshold
        print("\nMerging topics based on hierarchical clustering...")
        try:
            threshold = self.config.get("hierarchical_clustering_threshold", 1.5)
            from scipy.cluster.hierarchy import fcluster
            # Obtain topic embeddings and topic IDs, excluding -1
            topic_ids = sorted([topic_id for topic_id in self.topic_model.get_topics().keys() if topic_id != -1])
            topic_embeddings = []
            for topic_id in topic_ids:
                embedding = self.topic_model.topic_embeddings_[topic_id]
                topic_embeddings.append(embedding)
            topic_embeddings = np.array(topic_embeddings)
            # Compute the linkage matrix
            linkage_matrix = sch.linkage(topic_embeddings, method='ward', optimal_ordering=True)
            # Assign cluster labels based on the threshold
            cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
            # Map topic IDs to cluster labels
            topic_id_to_cluster_label = dict(zip(topic_ids, cluster_labels))
            # Create a mapping from cluster labels to topic IDs
            from collections import defaultdict
            cluster_to_topic_ids = defaultdict(list)
            for topic_id, cluster_label in topic_id_to_cluster_label.items():
                cluster_to_topic_ids[cluster_label].append(topic_id)
            # Merge topics in each cluster
            for cluster_label, topic_ids in cluster_to_topic_ids.items():
                if len(topic_ids) > 1:
                    print(f"Merging topics {topic_ids} into one topic (cluster {cluster_label})")
                    try:
                        self.topic_model.merge_topics(
                            self.docs,
                            topics_to_merge=topic_ids
                        )
                        # Update topics and embeddings
                        self.topic_model.update_topics(self.docs)
                    except Exception as e:
                        print(f"An error occurred while merging topics {topic_ids}: {e}")
            print("Topic merging based on hierarchical clustering completed.")

            # Verify the updated topic information
            print("\nGetting updated topic information after merging...")
            topic_info_after_merging = self.topic_model.get_topic_info()
            print(topic_info_after_merging)

        except Exception as e:
            print(f"An error occurred during topic merging: {e}")
            import traceback
            traceback.print_exc()

        # After merging, generate hierarchical topics again
        print("\nGenerating hierarchical topics after merging...")
        try:
            hierarchical_topics_after = self.topic_model.hierarchical_topics(
                self.docs,
                linkage_function=lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
            )
            hierarchical_topics_output_path_after = os.path.join(self.config.get("output_dir", "."), 'hierarchical_topics_after_merging.csv')
            hierarchical_topics_after.to_csv(hierarchical_topics_output_path_after, index=False)
            print(f"Hierarchical topics after merging saved to {hierarchical_topics_output_path_after}.")
        except Exception as e:
            print(f"An error occurred while generating hierarchical topics after merging: {e}")
            import traceback
            traceback.print_exc()

        # Save visualize_hierarchy after merging
        print("\nVisualizing hierarchy after merging...")
        try:
            hierarchy_fig_after = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics_after)
            hierarchy_fig_after.write_html(os.path.join(self.config.get("output_dir", "."), 'topic_hierarchy_after_merging.html'))
            print("Hierarchy visualization after merging saved.")
        except Exception as e:
            print(f"An error occurred while visualizing hierarchy after merging: {e}")
            import traceback
            traceback.print_exc()


    def _customize_topic_labels(self):
        """Customize topic labels to include zero-shot topic names followed by top words."""
        # Get zero-shot topic list and seed words from config
        zeroshot_topic_list = self.config.get("zeroshot_topic_list", [])
        seed_words_dict = self.config.get("seed_words", {})

        # Create a mapping from seed words to their respective topics
        seed_word_to_topic = {}
        for topic_name, words in seed_words_dict.items():
            for word in words:
                seed_word_to_topic[word] = topic_name

        # Initialize a dictionary to hold the new labels
        topic_labels = {}

        # Get topic info from the model
        topic_info = self.topic_model.get_topic_info()

        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                topic_labels[topic_id] = "Outliers"
            else:
                # Get the top words for the topic
                top_words = [word for word, _ in self.topic_model.get_topic(topic_id)]
                # Find if any seed word is present in the top words
                matched_topic = None
                for word in top_words:
                    if word in seed_word_to_topic:
                        matched_topic = seed_word_to_topic[word]
                        break  # Assign the first matching topic

                if matched_topic:
                    label = matched_topic
                else:
                    label = f"Topic {topic_id}"
                # Combine the topic name with the top words
                top_words_str = ', '.join(top_words[:10])  # Limit to top 10 words
                topic_labels[topic_id] = f"{label}: {top_words_str}"

        # Update the topic labels in the model
        self.topic_model.set_topic_labels(topic_labels)

    def merge_similar_topics(self, topic_list):
        """
        Merge topics in the topic model that are similar to the topics in topic_list,
        based on a similarity threshold.

        The method finds topics similar to each topic in topic_list and merges them together
        if their similarity score exceeds the threshold. Once a topic is merged, it can still
        be considered for merging with other topics in subsequent iterations.
        """
        current_model = self.topic_model

        for topic_name in topic_list:
            print(f"\nProcessing topic: '{topic_name}'")
            # Find topics similar to topic_name
            similar_topics, similarities = current_model.find_topics(search_term=topic_name, top_n=10)
            topics_to_merge = []
            for topic_id, sim in zip(similar_topics, similarities):
                if topic_id != -1 and sim >= self.similarity_threshold:
                    topics_to_merge.append((topic_id, sim))

            if len(topics_to_merge) > 1:
                # Sort topics by similarity in descending order
                topics_to_merge.sort(key=lambda x: x[1], reverse=True)
                # Merge topics sequentially
                while len(topics_to_merge) > 1:
                    base_topic_id, base_sim = topics_to_merge.pop(0)
                    next_topic_id, next_sim = topics_to_merge.pop(0)
                    print(f"\nMerging Topic {base_topic_id} and Topic {next_topic_id}")

                    # Merge the topics
                    current_model.merge_topics(
                        self.docs,
                        topics_to_merge=[base_topic_id, next_topic_id],
                        embeddings=self.embeddings
                    )

                    # Update topic representations
                    current_model.update_topics(self.docs, embeddings=self.embeddings)
                    print("Topics merged successfully.")

                    # Re-fetch similar topics for the current topic_name
                    similar_topics, similarities = current_model.find_topics(search_term=topic_name, top_n=10)
                    topics_to_merge = []
                    for topic_id, sim in zip(similar_topics, similarities):
                        if topic_id != -1 and sim >= self.similarity_threshold:
                            topics_to_merge.append((topic_id, sim))
                    # Sort topics again
                    topics_to_merge.sort(key=lambda x: x[1], reverse=True)
            elif len(topics_to_merge) == 1:
                print(f"Only one topic found similar to '{topic_name}' with similarity above threshold. No merging performed.")
            else:
                print(f"No similar topics found for '{topic_name}' with similarity above threshold.")

        # Update the topic model
        self.topic_model = current_model

    def save_topic_info(self):
        """Save topic information to a CSV file."""
        topic_info = self.topic_model.get_topic_info()
        output_file = os.path.join(self.config.get("output_dir", "."), "topic_info.csv")
        topic_info.to_csv(output_file, index=False)
        print(f"Topic information saved to {output_file}.")

    def generate_additional_visualizations(self):
        """
        Generate and save additional visualizations such as topics, topic hierarchy, term rank, etc.
        """
        output_dir = self.config.get("output_dir", ".")

        print("Generating additional visualizations...")
        try:
            # Visualize topics
            topics_fig = self.topic_model.visualize_topics()
            topics_fig.write_html(os.path.join(output_dir, 'topics.html'))

            # Visualize hierarchical topics
            hierarchy_fig = self.topic_model.visualize_hierarchy()
            hierarchy_fig.write_html(os.path.join(output_dir, 'topic_hierarchy.html'))

            # Visualize topic terms
            barchart_fig = self.topic_model.visualize_barchart(top_n_topics=50)
            barchart_fig.write_html(os.path.join(output_dir, 'topic_barchart.html'))

            # Visualize topic similarity
            heatmap_fig = self.topic_model.visualize_heatmap()
            heatmap_fig.write_html(os.path.join(output_dir, 'topic_heatmap.html'))

            # Visualize term rank
            term_rank_fig = self.topic_model.visualize_term_rank()
            term_rank_fig.write_html(os.path.join(output_dir, 'term_rank.html'))

            # Visualize topics over time if you have timestamps
            # timestamps = ...  # Provide your timestamps here
            # topics_over_time = self.topic_model.topics_over_time(self.docs, timestamps)
            # topics_over_time_fig = self.topic_model.visualize_topics_over_time(topics_over_time)
            # topics_over_time_fig.write_html(os.path.join(output_dir, 'topics_over_time.html'))

            print("Additional visualizations generated and saved.")
        except Exception as e:
            print(f"An error occurred while generating visualizations: {e}")
            import traceback
            traceback.print_exc()

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

    # Load configuration variables from config.json with fallback path
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            print("Config File Loaded from 'config.json'.")
    except FileNotFoundError:
        fallback_config_path = r"C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config_hlr.json"
        with open(fallback_config_path, 'r') as config_file:
            config = json.load(config_file)
            print(f"Config File Loaded from fallback path: {fallback_config_path}")
            
    # Set random seed
    random_seed = config.get("random_seed", 42)
    np.random.seed(random_seed)

    # Extract variables from the config
    sample_size = config.get("sample_size", 1000)
    document_split = config.get("document_split", "default_method")
    section_to_analyze = config.get("section_to_analyze", "default_section")
    max_documents = config.get("max_documents", 1000)
    # Removed similarity_threshold from here since it's now handled in the class

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

    # Train the model
    bertopic_model.train(all_relevant_sections)

    # Generate additional visualizations
    bertopic_model.generate_additional_visualizations()

    print("BERTopic model training and saving completed.")

    # Save topic information to CSV
    #bertopic_model.save_topic_info()  # Optional: save topic information

    # End total execution time tracking
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total execution time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()
