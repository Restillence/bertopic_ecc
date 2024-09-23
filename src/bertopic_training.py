import os

# Set TOKENIZERS_PARALLELISM to 'false' to disable parallelism and suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import numpy as np
import torch  # For checking if GPU is available
import threading  # For Heartbeat functionality
import GPUtil  # For GPU monitoring
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from utils import print_configuration
from transformers import pipeline, AutoTokenizer

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
        self.model_save_path = config["model_save_path"]
        self.modeling_type = config.get("modeling_type", "regular")  # Options: ["regular", "iterative", "iterative_zeroshot", "zeroshot"]
        self.doc_chunk_size = config.get("doc_chunk_size", 5000)  # Used for iterative training
        self.device = self._select_device()
        self.topic_model = None
        self.batch_size = config.get("batch_size", 32)
        self.nr_topics = config.get("nr_topics", None)  # Added nr_topics
        self.model = self._select_embedding_model(config)

    def _select_device(self):
        """Check if GPU is available and return the correct device."""
        if torch.cuda.is_available():
            print("GPU is available. Using GPU...")
            return torch.device("cuda")
        else:
            print("GPU not available. Falling back to CPU...")
            return torch.device("cpu")

    def _select_embedding_model(self, config):
        """Select the embedding model based on the config setting."""
        embedding_choice = config.get("embedding_model_choice", "all-MiniLM-L12-v2")
        
        if embedding_choice == "all-MiniLM-L12-v2":
            print("Loading SentenceTransformer model: all-MiniLM-L12-v2...")
            return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=self.device)
        
        elif embedding_choice == "finbert-local":
            model_path = config["finbert_model_path"]
            print(f"Loading FinBERT model from local path: {model_path} on {self.device}...")
            if not os.path.exists(model_path):
                raise ValueError(f"The specified model path does not exist: {model_path}")
            return SentenceTransformer(model_path, device=self.device)

        elif embedding_choice == "finbert-pretrain":
            print("Loading FinBERT model from HuggingFace pipeline...")
            return self._load_finbert_pipeline()
        
        else:
            raise ValueError(f"Unknown embedding model choice: {embedding_choice}")

    def _load_finbert_pipeline(self):
        """Load the FinBERT model from the Hugging Face pipeline."""
        print(f"Loading FinBERT model pipeline from HuggingFace on {self.device}...")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
        
        # Force tokenizer to truncate inputs at 512 tokens
        tokenizer.model_max_length = 512
        tokenizer.truncation = True

        # Set up the pipeline with the model and tokenizer
        pipe = pipeline(
            "feature-extraction",
            model="yiyanghkust/finbert-pretrain",
            tokenizer=tokenizer,
            device=0 if self.device.type == "cuda" else -1
        )
        return pipe

    def _initialize_bertopic_model(self):
        """Initialize the BERTopic model with the specified parameters."""
        print(f"Embedding Model used: {self.model}...")
        # Initialize CountVectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=tuple(self.config["vectorizer_model_params"]["ngram_range"]),
            stop_words=self.config["vectorizer_model_params"]["stop_words"],
            min_df=self.config["vectorizer_model_params"]["min_df"]
        )

        # Initialize UMAP
        umap_model = UMAP(
            n_neighbors=self.config["umap_model_params"]["n_neighbors"],
            n_components=self.config["umap_model_params"]["n_components"],
            min_dist=self.config["umap_model_params"]["min_dist"],
            metric=self.config["umap_model_params"]["metric"],
            random_state=42
        )

        # Initialize HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config["hdbscan_model_params"]["min_cluster_size"],
            metric=self.config["hdbscan_model_params"]["metric"],
            cluster_selection_method=self.config["hdbscan_model_params"]["cluster_selection_method"],
            prediction_data=self.config["hdbscan_model_params"]["prediction_data"]
        )

        # Initialize KeyBERTInspired and MaximalMarginalRelevance using the parameters from config
        keybert_model = KeyBERTInspired(top_n_words=self.config["keybert_params"]["top_n_words"])
        mmr_model = MaximalMarginalRelevance(diversity=self.config["mmr_params"]["diversity"])

        # Initialize BERTopic with both representation models (KeyBERTInspired and MaximalMarginalRelevance)
        if self.modeling_type in ["zeroshot", "iterative_zeroshot"]:
            print("Initializing zeroshot BERTopic model...")
            return BERTopic(
                embedding_model=self.model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                zeroshot_topic_list=self.config["zeroshot_topic_list"],
                zeroshot_min_similarity=self.config["zeroshot_min_similarity"],
                representation_model=[keybert_model, mmr_model],
                min_topic_size=self.config.get("min_topic_size", 15)  # Ensure smaller topics can be captured
            )
        else:
            print("Initializing regular BERTopic model...")
            return BERTopic(
                embedding_model=self.model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=[keybert_model, mmr_model],  # Combined representation
                min_topic_size=self.config.get("min_topic_size", 15)  # Ensure smaller topics can be captured
            )

    def _heartbeat(self, stop_event, interval=300):
        """Periodically print a heartbeat message and GPU usage to keep the connection alive.

        Parameters
        ----------
        stop_event : threading.Event
            Event to signal the thread to stop.
        interval : int
            Time interval in seconds between heartbeat messages. Default is 300 (5 minutes).

        Returns
        -------
        None
        """
        while not stop_event.is_set():
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))
            heartbeat_message = f"[Heartbeat] Still working... Time elapsed: {elapsed_time}"
            
            # Get GPU usage if GPU is available
            if self.device.type == "cuda":
                gpus = GPUtil.getGPUs()
                gpu_status = []
                for gpu in gpus:
                    gpu_status.append(
                        f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}MB/{gpu.memoryTotal}MB memory"
                    )
                gpu_message = " | ".join(gpu_status)
                full_message = f"{heartbeat_message} | {gpu_message}"
            else:
                full_message = heartbeat_message

            print(full_message)
            stop_event.wait(interval)

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
        # Check if the modeling type is iterative or iterative_zeroshot
        if self.modeling_type in ["iterative", "iterative_zeroshot"]:
            # Train the model using the iterative approach
            self._train_iterative(docs)
        else:
            # Train the model using the regular approach
            self._train_regular(docs)

    def _train_regular(self, docs):
        # Initialize BERTopic model
        self.topic_model = self._initialize_bertopic_model()

        # Start timer
        self.start_time = time.time()

        # Initialize the stop event for the heartbeat thread
        stop_event = threading.Event()

        # Start the heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat, args=(stop_event,))
        heartbeat_thread.daemon = True  # Ensures the thread exits when the main program does
        heartbeat_thread.start()

        # Compute embeddings with reduced batch size
        print(f"Computing embeddings with batch size {self.batch_size}...")
        embeddings = self.model.encode(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        # Train the BERTopic model
        print(f"Training BERTopic model using the following modeling type: {self.modeling_type}...")
        try:
            topics, probs = self.topic_model.fit_transform(docs, embeddings)

            # Reduce the number of topics to the desired number
            if self.nr_topics is not None:
                print(f"Reducing the number of topics to {self.nr_topics}...")
                self.topic_model.reduce_topics(
                    docs=docs,
                    nr_topics=self.nr_topics
                )
                topics = self.topic_model.topics_
                probs = self.topic_model.probabilities_

            # Handle None values in topics (assign -1 to unassigned topics)
            topics = [topic if topic is not None else -1 for topic in topics]

            # Print document count per topic for debugging
            topic_doc_count = {topic: topics.count(topic) for topic in set(topics)}
            print("Document count per topic:", topic_doc_count)

            # Assign topics and probabilities to the model
            self.topic_model.topics_ = topics
            self.topic_model.probabilities_ = probs
            self.topic_model.original_documents_ = docs  # Ensure original_documents_ is set

        except Exception as e:
            print(f"An error occurred during model training: {e}")
            # Stop the heartbeat thread in case of an error
            stop_event.set()
            heartbeat_thread.join()
            return

        # Stop the heartbeat thread after training completes
        stop_event.set()
        heartbeat_thread.join()

        # End timer
        end_time = time.time()

        # Print information about the training process
        print(f"BERTopic model trained on {len(docs)} sections.")
        print(f"Number of topics generated: {len(set(topics))}")
        print(f"Training time: {end_time - self.start_time:.2f} seconds.")

        # Save the BERTopic model using safetensors
        try:
            print("Saving BERTopic model...")
            embedding_model = self.config["finbert_model_path"]
            self.topic_model.save(
                self.model_save_path,
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=embedding_model
            )
            print(f"BERTopic model saved to {self.model_save_path}.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def _train_iterative(self, docs):
        print("Initializing iterative BERTopic model...")

        # Split the input documents into chunks
        doc_chunks = [docs[i:i+self.doc_chunk_size] for i in range(0, len(docs), self.doc_chunk_size)]

        if not doc_chunks:
            print("No document chunks found for iterative training.")
            return

        # Initialize the base model with the first chunk of documents
        self.topic_model = self._initialize_bertopic_model()

        # Compute embeddings for the first chunk
        print(f"Computing embeddings for the first chunk with batch size {self.batch_size}...")
        embeddings_chunk = self.model.encode(
            doc_chunks[0],
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        base_model = self.topic_model.fit(doc_chunks[0], embeddings_chunk)
        base_model.original_documents_ = doc_chunks[0]

        # Start timer
        self.start_time = time.time()

        # Initialize the stop event for the heartbeat thread
        stop_event = threading.Event()

        # Start the heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat, args=(stop_event,))
        heartbeat_thread.daemon = True  # Ensures the thread exits when the main program does
        heartbeat_thread.start()

        # Iterate over the remaining chunks of documents
        for chunk in doc_chunks[1:]:
            print("Merging new documents into the base model...")

            try:
                # Compute embeddings for the current chunk
                print(f"Computing embeddings for the current chunk with batch size {self.batch_size}...")
                embeddings_chunk = self.model.encode(
                    chunk,
                    batch_size=self.batch_size,
                    show_progress_bar=True
                )

                # Train a new model on the current chunk of documents
                new_model = self._initialize_bertopic_model().fit(chunk, embeddings_chunk)
                new_model.original_documents_ = chunk

                # Merge the new model with the base model
                updated_model = BERTopic.merge_models([base_model, new_model])

                # Print the number of newly discovered topics
                nr_new_topics = len(set(updated_model.topics_)) - len(set(base_model.topics_))
                new_topics = list(updated_model.topic_labels_.values())[-nr_new_topics:]
                print("The following topics are newly found:")
                print(f"{new_topics}\n")

                # Update the base model
                base_model = updated_model

            except Exception as e:
                print(f"An error occurred during iterative training: {e}")
                # Stop the heartbeat thread in case of an error
                stop_event.set()
                heartbeat_thread.join()
                return

        # Assign the final merged model
        self.topic_model = base_model

        # Reduce the number of topics to the desired number
        if self.nr_topics is not None:
            print(f"Reducing the number of topics to {self.nr_topics}...")
            # Combine all documents and embeddings
            all_docs = []
            all_embeddings = []
            for chunk in doc_chunks:
                all_docs.extend(chunk)
                embeddings_chunk = self.model.encode(
                    chunk,
                    batch_size=self.batch_size,
                    show_progress_bar=True
                )
                all_embeddings.extend(embeddings_chunk)
            # Reduce topics
            self.topic_model.reduce_topics(
                docs=all_docs,
                nr_topics=self.nr_topics
            )

        # Stop the heartbeat thread after training completes
        stop_event.set()
        heartbeat_thread.join()

        # End timer
        end_time = time.time()

        # Print information about the training process
        print(f"Iterative BERTopic model trained on {len(docs)} sections.")
        print(f"Number of topics generated after reduction: {len(set(self.topic_model.topics_))}")
        print(f"Training time: {end_time - self.start_time:.2f} seconds.")

        # Save the final merged model
        try:
            print("Saving the final merged BERTopic model using safetensors...")
            embedding_model = self.config["finbert_model_path"]
            self.topic_model.save(
                self.model_save_path,
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=embedding_model
            )
            print(f"Final BERTopic model saved to {self.model_save_path}.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

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

    # Initialize FileHandler and TextProcessor with the imported configuration
    print("Initializing file handler and text processor...")
    file_handler = FileHandler(index_file_path=config["index_file_path"], folderpath_ecc=folderpath_ecc)
    text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

    # Create the sample and extract relevant sections
    print("Reading index file and creating ECC sample...")
    index_file = file_handler.read_index_file()
    ecc_sample = file_handler.create_ecc_sample(sample_size)
    all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)

    if not all_relevant_sections:
        print("No relevant sections found to fit BERTopic.")
        return

    # Instantiate and train the BERTopic model
    bertopic_model = BertopicModel(config)
    bertopic_model.train(all_relevant_sections)

    print("BERTopic model training and saving completed.")

if __name__ == "__main__":
    main()
