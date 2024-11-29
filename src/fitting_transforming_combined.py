import os
import sys
import json
import time
import threading
import datetime
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler
from text_processing import TextProcessor
from utils import print_configuration
from sentence_transformers import SentenceTransformer
from scipy.cluster import hierarchy as sch

# Disable parallelism in tokenizers to prevent CPU overutilization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Adjust the path to include 'src' if it's not already in the system path
current_dir = os.getcwd()
if "src" not in current_dir:
    src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
    sys.path.append(src_path)

# Import the evaluation module
from evaluate_topics import generate_evaluation_file

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
        from sklearn.feature_extraction.text import CountVectorizer
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

    def _post_training_tasks(self):
        """Perform tasks after training, such as merging topics and customizing labels."""
        # Check if topic merging is enabled in the configuration
        if self.apply_topic_merging:
            # Proceed to merge similar topics based on the topic_list
            topic_list = [
                "regulation and compliance",
                "risk and forecasts",
                "competition and strategy",
                "consumer and demand",
                "economy",
                "revenue and sales",
                "products and services",
                "earnings and income",
                "operations and management",
                "investments and capital",
                "geography and regions",
                "growth and strategy",
                "tax and policies",
                "expenses and costs",
                "marketing and advertising"
            ]

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

class CustomEmbeddingModel(BaseEmbedder):
    """
    A wrapper class for the SentenceTransformer model to add the 'embed_documents' method
    expected by BERTopic.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, documents, verbose=False):
        return self.embedding_model.encode(documents, show_progress_bar=verbose)

    def embed_queries(self, queries, verbose=False):
        return self.embedding_model.encode(queries, show_progress_bar=verbose)

def save_results(topics_sections, topics_questions, topics_answers, ecc_sample, all_relevant_sections, all_relevant_questions, all_management_answers, config):
    """
    Save the results to a CSV file and return the DataFrame.
    """
    result_dict = {}
    topic_idx_sections = 0
    topic_idx_questions = 0
    topic_idx_answers = 0

    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            sections = value.get('presentation_text', [])
            num_sections = len(sections)
            section_topics = topics_sections[topic_idx_sections:topic_idx_sections + num_sections]

            questions = value.get('participant_questions', [])
            num_questions = len(questions)
            question_topics = topics_questions[topic_idx_questions:topic_idx_questions + num_questions]

            answers = value.get('management_answers', [])
            num_answers = len(answers)
            answer_topics = topics_answers[topic_idx_answers:topic_idx_answers + num_answers]

            # Convert topics to list if they are numpy arrays
            if isinstance(section_topics, np.ndarray):
                section_topics = section_topics.tolist()
            if isinstance(question_topics, np.ndarray):
                question_topics = question_topics.tolist()
            if isinstance(answer_topics, np.ndarray):
                answer_topics = answer_topics.tolist()

            # Get the timestamp for the call
            timestamp = value.get('date', 'Unknown')

            # Get company_info
            company_info = value.get('company_info', 'Unknown')

            # Get ceo_participates flag
            ceo_participates = value.get('ceo_participates', False)

            # Get CEO and CFO names
            ceo_names = value.get('ceo_names', [])
            cfo_names = value.get('cfo_names', [])

            # Store in result_dict
            result_dict[call_id] = {
                "permco": permco,
                "company_info": company_info,
                "date": timestamp,
                "presentation_text": sections,
                "presentation_topics": section_topics,
                "participant_questions": questions,
                "participant_question_topics": question_topics,
                "management_answers": answers,
                "management_answer_topics": answer_topics,
                "ceo_participates": int(ceo_participates),  # Convert bool to int
                "ceo_names": ceo_names,
                "cfo_names": cfo_names
            }
            topic_idx_sections += num_sections
            topic_idx_questions += num_questions
            topic_idx_answers += num_answers

    # Convert results to DataFrame
    records = []
    for call_id, call_data in result_dict.items():
        records.append({
            'permco': call_data['permco'],
            'call_id': call_id,
            'company_info': call_data['company_info'],
            'date': call_data['date'],
            'presentation_text': json.dumps(call_data['presentation_text']),
            'presentation_topics': json.dumps(call_data['presentation_topics']),
            'participant_questions': json.dumps(call_data['participant_questions']),
            'participant_question_topics': json.dumps(call_data['participant_question_topics']),
            'management_answers': json.dumps(call_data['management_answers']),
            'management_answer_topics': json.dumps(call_data['management_answer_topics']),
            'ceo_participates': int(call_data['ceo_participates']),  # Convert bool to int
            'ceo_names': json.dumps(call_data['ceo_names']),
            'cfo_names': json.dumps(call_data['cfo_names'])
        })

    results_df = pd.DataFrame(records)
    results_output_path = os.path.join(config.get("index_file_ecc_folder", "."), 'topics_output_combined.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}.")

    return results_df

def save_visualization(fig, output_file, file_format="png"):
    """
    Save a visualization from BERTopic's built-in visualizations.

    Args:
        fig: The figure to save.
        output_file: The path to save the figure.
        file_format: The format to save the figure ('png' or 'html').
    """
    start_time = time.time()
    if file_format == "html":
        fig.write_html(output_file)
    else:
        fig.write_image(output_file)
    end_time = time.time()
    print(f"Visualization saved to {output_file} in {end_time - start_time:.2f} seconds.")

def visualize_topics_over_time(topic_model, results_df, output_dir):
    """
    Generate and save the Topics over Time visualization.
    """
    try:
        start_time = time.time()

        # Prepare the data
        timestamps = []
        documents = []
        topics_list = []

        for index, row in results_df.iterrows():
            date = row['date']  # The date of the conference call

            # For presentation texts
            sections = json.loads(row['presentation_text'])  # List of sections (paragraphs)
            section_topics = json.loads(row['presentation_topics'])  # List of topics for the sections

            num_sections = len(sections)
            timestamps.extend([date] * num_sections)
            documents.extend(sections)
            topics_list.extend(section_topics)

            # For participant questions
            questions = json.loads(row['participant_questions'])
            question_topics = json.loads(row['participant_question_topics'])

            num_questions = len(questions)
            timestamps.extend([date] * num_questions)
            documents.extend(questions)
            topics_list.extend(question_topics)

            # For management answers
            answers = json.loads(row['management_answers'])
            answer_topics = json.loads(row['management_answer_topics'])

            num_answers = len(answers)
            timestamps.extend([date] * num_answers)
            documents.extend(answers)
            topics_list.extend(answer_topics)

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(timestamps)

        # Ensure that the number of timestamps matches the number of documents and topics
        if not (len(timestamps) == len(documents) == len(topics_list)):
            raise ValueError("Number of timestamps, documents, and topics do not match.")

        # Set the number of bins to a value lower than 100
        nr_bins = 50  # Adjust as needed

        # Generate topics over time with the specified number of bins
        topics_over_time = topic_model.topics_over_time(
            docs=documents,
            topics=topics_list,
            timestamps=timestamps,
            nr_bins=nr_bins
        )

        # Check if topics_over_time is empty
        if topics_over_time.empty:
            print("No data available for topics over time visualization.")
            return

        # Visualize topics over time with normalize_frequency=True
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=10,
            normalize_frequency=True
        )
        save_visualization(
            fig,
            os.path.join(output_dir, "topics_over_time.html"),
            file_format="html"
        )

        end_time = time.time()
        print(f"Topics over time visualization saved in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred in visualize_topics_over_time: {e}")
        import traceback
        traceback.print_exc()

def generate_additional_visualizations(topic_model, results_df, output_dir):
    """
    Generate and save additional visualizations.
    """
    print("Generating additional visualizations...")
    start_time = time.time()

    # Visualize Topics
    print("Visualizing topics...")
    fig = topic_model.visualize_topics()
    save_visualization(fig, os.path.join(output_dir, "topics.html"), file_format="html")

    # Visualize Topic Hierarchy
    print("Visualizing topic hierarchy...")
    fig = topic_model.visualize_hierarchy()
    save_visualization(fig, os.path.join(output_dir, "topic_hierarchy.html"), file_format="html")

    # Visualize Topic Terms (BarChart)
    print("Visualizing topic terms...")
    fig = topic_model.visualize_barchart()
    save_visualization(fig, os.path.join(output_dir, "topic_barchart.html"), file_format="html")

    # Visualize Topic Similarity (Heatmap)
    print("Visualizing topic similarity...")
    fig = topic_model.visualize_heatmap()
    save_visualization(fig, os.path.join(output_dir, "topic_heatmap.html"), file_format="html")

    # Visualize Term Score Decline (Term Rank)
    print("Visualizing term rank...")
    fig = topic_model.visualize_term_rank()
    save_visualization(fig, os.path.join(output_dir, "term_rank.html"), file_format="html")

    # Visualize Topics over Time
    #print("Visualizing topics over time...")
    #visualize_topics_over_time(topic_model, results_df, output_dir)
    print("skipping topics over time visualization...")

    end_time = time.time()
    print(f"All visualizations generated and saved in {end_time - start_time:.2f} seconds.")

def generate_evaluation_files(topic_model, results_df, eval_dir):
    """
    Generate evaluation files for Presentation, Q&A, and Management Answer sections.
    """
    # Define the sections and their corresponding columns
    sections = [
        {
            'section_type': 'Presentation',
            'text_column': 'presentation_text',
            'topics_column': 'presentation_topics'
        },
        {
            'section_type': 'Participant Questions',
            'text_column': 'participant_questions',
            'topics_column': 'participant_question_topics'
        },
        {
            'section_type': 'Management Answers',
            'text_column': 'management_answers',
            'topics_column': 'management_answer_topics'
        }
    ]

    for section in sections:
        print(f"Generating evaluation file for {section['section_type']} sections...")
        generate_evaluation_file(
            topic_model=topic_model,
            results_df=results_df,
            output_dir=eval_dir,
            text_column=section['text_column'],
            topics_column=section['topics_column'],
            section_type=section['section_type']
        )
        print(f"Evaluation file for {section['section_type']} saved.\n")

def generate_visualizations_and_evaluation_files(topic_model, results_df, config):
    """
    Generate visualizations and evaluation files.
    """
    output_dir = config.get("output_dir", "model_outputs")
    eval_dir = config.get("eval_dir", "eval")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Generate additional visualizations
    generate_additional_visualizations(topic_model, results_df, output_dir)

    # Generate evaluation files
    generate_evaluation_files(topic_model, results_df, eval_dir)

def heartbeat():
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Heartbeat: The current time is {current_time}")
        time.sleep(300)  # Sleep for 5 minutes (300 seconds)

def main():
    """
    Main entry point of the script.
    """
    # Start the heartbeat thread
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    # Start total execution time tracking
    total_start_time = time.time()

    # Load configuration from config.json
    print("Loading configuration...")
    # Define the relative path to config.json
    relative_path = 'config_hlr.json'

    # Define the fallback absolute path
    fallback_path = r'C:\Users\nikla\OneDrive\Dokumente\winfoMaster\Masterarbeit\bertopic_ecc\config.json'

    try:
        # Attempt to load config.json from the relative path
        with open(relative_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from relative path: {os.path.abspath(relative_path)}")
    except FileNotFoundError:
        print(f"Relative config.json not found at {os.path.abspath(relative_path)}. Trying fallback path...")
        try:
            # Attempt to load config.json from the absolute path
            with open(fallback_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from fallback path: {fallback_path}")
        except FileNotFoundError:
            # Handle the case where both paths fail
            print("Failed to load configuration. config.json not found in both relative and fallback paths.")
            config = {}

    # Set random seed
    random_seed = config.get("random_seed", 42)
    np.random.seed(random_seed)
    print_configuration(config)

    # Initialize FileHandler and TextProcessor with the imported configuration
    print("Initializing file handler and text processor...")
    file_handler = FileHandler(config=config)
    text_processor = TextProcessor(method=config.get("document_split", "default_method"))
    # Removed 'section_to_analyze' parameter

    # Read index file and create ECC sample
    print("Reading index file and creating ECC sample...")
    index_file = file_handler.read_index_file()
    sample_size = config.get("sample_size", 1000)
    ecc_sample = file_handler.create_ecc_sample(sample_size)

    # Extract relevant sections
    print("Extracting and processing relevant sections...")
    extraction_start_time = time.time()
    all_relevant_sections = []
    all_relevant_questions = []
    all_management_answers = []
    ecc_sample_filtered = {}
    not_considered_count = 0  # Initialize counter

    for permco, calls in ecc_sample.items():
        calls_filtered = {}
        for call_id, value in calls.items():
            company_info = value.get('company_name', 'Unknown')
            date = value.get('date', 'Unknown')
            text = value.get('text_content', '')
            result = text_processor.extract_and_split_section(permco, call_id, company_info, date, text)
            if result and (result.get('presentation_text') or result.get('participant_questions') or result.get('management_answers')):
                # Process presentation_text
                if result.get('presentation_text'):
                    all_relevant_sections.extend(result['presentation_text'])
                # Process participant_questions
                if result.get('participant_questions'):
                    all_relevant_questions.extend(result['participant_questions'])
                # Process management_answers
                if result.get('management_answers'):
                    all_management_answers.extend(result['management_answers'])
                # Add the relevant data to 'value'
                value['presentation_text'] = result.get('presentation_text', [])
                value['participant_questions'] = result.get('participant_questions', [])
                value['management_answers'] = result.get('management_answers', [])
                value['participants'] = result.get('participants', [])
                value['ceo_participates'] = result.get('ceo_participates', False)
                value['ceo_names'] = result.get('ceo_names', [])
                value['cfo_names'] = result.get('cfo_names', [])
                # Add 'company_info' to 'value'
                value['company_info'] = company_info
                # Add the call to calls_filtered
                calls_filtered[call_id] = value
            else:
                print(f"Earnings call {call_id} has no relevant sections and will be excluded.")
                not_considered_count += 1
        if calls_filtered:
            ecc_sample_filtered[permco] = calls_filtered
    extraction_end_time = time.time()
    print(f"Extraction and processing completed in {extraction_end_time - extraction_start_time:.2f} seconds.")
    print(f"Total number of earnings calls not considered due to missing sections: {not_considered_count}")

    if not all_relevant_sections and not all_relevant_questions and not all_management_answers:
        print("No relevant sections or participant questions or management answers found to fit BERTopic.")
        return

    # Combine all documents
    all_documents = all_relevant_sections + all_relevant_questions + all_management_answers
    num_sections = len(all_relevant_sections)
    num_questions = len(all_relevant_questions)
    num_answers = len(all_management_answers)

    print(f"Total documents: {len(all_documents)}")
    print(f"Number of sections: {num_sections}")
    print(f"Number of questions: {num_questions}")
    print(f"Number of answers: {num_answers}")

    # Initialize BertopicModel
    bertopic_model = BertopicModel(config)

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = bertopic_model.model.encode(all_documents, show_progress_bar=True, batch_size=config.get("batch_size", 64))

    # Set bertopic_model.docs and embeddings
    bertopic_model.docs = all_documents
    bertopic_model.embeddings = embeddings

    # Initialize BERTopic model
    bertopic_model.topic_model = bertopic_model._initialize_bertopic_model()

    # Train the model using fit_transform
    print("Training the BERTopic model using fit_transform...")
    topics, probabilities = bertopic_model.topic_model.fit_transform(all_documents, embeddings)

    # Perform post-training tasks
    bertopic_model._post_training_tasks()

    # Save the model
    try:
        print("Saving BERTopic model...")
        bertopic_model.topic_model.save(
            bertopic_model.model_save_path,
            serialization="safetensors",
            save_ctfidf=True
        )
        print(f"BERTopic model saved to {bertopic_model.model_save_path}.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

    # Split topics and probabilities
    topics_sections = topics[:num_sections]
    probabilities_sections = probabilities[:num_sections]
    topics_questions = topics[num_sections:num_sections + num_questions]
    probabilities_questions = probabilities[num_sections:num_sections + num_questions]
    topics_answers = topics[num_sections + num_questions:]
    probabilities_answers = probabilities[num_sections + num_questions:]

    # Save results
    print("Saving results...")
    results_df = save_results(
        topics_sections,
        topics_questions,
        topics_answers,
        ecc_sample_filtered,
        all_relevant_sections,
        all_relevant_questions,
        all_management_answers,
        config
    )
    print("Results saved.")

    # Generate visualizations and evaluation files
    print("Generating visualizations and evaluation files...")
    generate_visualizations_and_evaluation_files(
        bertopic_model.topic_model,
        results_df,
        config
    )
    print("Visualizations and evaluation files generated.")

    # Total execution time
    total_end_time = time.time()
    print(f"Total script execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
