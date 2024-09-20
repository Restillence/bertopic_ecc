import os
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
import multiprocessing
from multiprocessing import Pool

# Worker function for embedding computation on a specific GPU
def compute_embeddings_worker(docs, embedding_choice, finbert_model_path, device_id):
    """
    Compute embeddings for a subset of documents on a specific GPU.

    Parameters
    ----------
    docs : list
        A list of strings representing the input documents.
    embedding_choice : str
        The choice of embedding model.
    finbert_model_path : str
        Path to the FinBERT model if using a local model.
    device_id : int
        The GPU device ID.

    Returns
    -------
    np.ndarray
        The computed embeddings.
    """
    if embedding_choice == "all-MiniLM-L12-v2":
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=f'cuda:{device_id}')
    elif embedding_choice == "finbert-local":
        if not os.path.exists(finbert_model_path):
            raise ValueError(f"The specified model path does not exist: {finbert_model_path}")
        model = SentenceTransformer(finbert_model_path, device=f'cuda:{device_id}')
    elif embedding_choice == "finbert-pretrain":
        # For finbert-pretrain, use the pipeline to compute embeddings
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
        tokenizer.model_max_length = 512
        tokenizer.truncation = True

        # Initialize the pipeline on the specified GPU
        pipe = pipeline(
            "feature-extraction",
            model="yiyanghkust/finbert-pretrain",
            tokenizer=tokenizer,
            device=device_id  # CUDA device ID
        )

        # Compute embeddings using the pipeline
        embeddings = []
        for doc in docs:
            features = pipe(doc)
            # Flatten the list of lists and convert to np.ndarray
            flat_features = np.array(features).flatten()
            embeddings.append(flat_features)
        return np.array(embeddings)
    else:
        raise ValueError(f"Unknown embedding model choice: {embedding_choice}")

    # Compute embeddings using the SentenceTransformer model
    embeddings = model.encode(docs, batch_size=32, show_progress_bar=False)
    return embeddings

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

        # Detect number of GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Multiple GPUs detected: {self.num_gpus} GPUs will be used.")
            self.devices = list(range(self.num_gpus))  # GPU IDs: 0, 1, ..., num_gpus-1
        elif self.num_gpus == 1:
            print("Single GPU detected. Using GPU 0.")
            self.devices = [0]
        else:
            print("No GPU detected. Using CPU.")
            self.devices = [-1]  # -1 indicates CPU

        self.topic_model = None

    def _select_embedding_model(self, config):
        """
        Select the embedding model based on the config setting.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        None
        """
        # The actual embedding computation will be handled separately
        pass  # Embedding models are loaded in worker processes

    def compute_embeddings(self, docs):
        """
        Compute embeddings using multiple GPUs if available.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        np.ndarray
            The computed embeddings.
        """
        embedding_choice = self.config.get("embedding_model_choice", "all-MiniLM-L12-v2")
        finbert_model_path = self.config.get("finbert_model_path", "")

        if self.devices == [-1]:
            # CPU mode
            print("Computing embeddings on CPU...")
            if embedding_choice == "finbert-pretrain":
                # Initialize pipeline
                tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
                tokenizer.model_max_length = 512
                tokenizer.truncation = True

                pipe = pipeline(
                    "feature-extraction",
                    model="yiyanghkust/finbert-pretrain",
                    tokenizer=tokenizer,
                    device=-1
                )

                embeddings = []
                for doc in docs:
                    features = pipe(doc)
                    flat_features = np.array(features).flatten()
                    embeddings.append(flat_features)
                return np.array(embeddings)
            else:
                model = SentenceTransformer(embedding_choice if embedding_choice != "finbert-local" else finbert_model_path, device='cpu')
                embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)
                return embeddings
        else:
            # GPU mode
            print(f"Computing embeddings on {len(self.devices)} GPU(s)...")
            # Split docs into chunks based on the number of GPUs
            chunks = np.array_split(docs, len(self.devices))

            # Prepare arguments for each worker
            args = []
            for i, chunk in enumerate(chunks):
                args.append((chunk.tolist(), embedding_choice, finbert_model_path, self.devices[i]))

            # Use multiprocessing Pool to compute embeddings in parallel
            with Pool(processes=len(self.devices)) as pool:
                results = pool.starmap(compute_embeddings_worker, args)

            # Concatenate all embeddings
            embeddings = np.vstack(results)
            return embeddings

    def _initialize_bertopic_model(self):
        """
        Initialize the BERTopic model with the specified parameters.

        Returns
        -------
        BERTopic
            An initialized BERTopic model.
        """
        print("Initializing BERTopic model...")
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
                embedding_model=None,  # Embeddings will be provided externally
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
                embedding_model=None,  # Embeddings will be provided externally
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=[keybert_model, mmr_model],  # Combined representation
                min_topic_size=self.config.get("min_topic_size", 15)  # Ensure smaller topics can be captured
            )

    def _heartbeat(self, stop_event, interval=900):
        """
        Periodically print a heartbeat message and GPU usage to keep the connection alive.

        Parameters
        ----------
        stop_event : threading.Event
            Event to signal the thread to stop.
        interval : int
            Time interval in seconds between heartbeat messages. Default is 900 (15 minutes).

        Returns
        -------
        None
        """
        while not stop_event.is_set():
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))
            heartbeat_message = f"[Heartbeat] Still working... Time elapsed: {elapsed_time}"
            
            # Get GPU usage if GPU is available
            if self.devices != [-1]:
                gpu_status = []
                for device_id in self.devices:
                    gpu = GPUtil.getGPUs()[device_id]
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
        """
        Train the BERTopic model using the specified modeling type.

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
        """
        Train the BERTopic model using the regular approach.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        None
        """
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

        # Compute embeddings
        print("Computing embeddings...")
        embeddings = self.compute_embeddings(docs)

        # Train the BERTopic model
        print(f"Training BERTopic model using the following modeling type: {self.modeling_type}...")
        try:
            topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)

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
            print(f"An error occurred during BERTopic training: {e}")
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
        print(f"BERTopic model trained on {len(docs)} documents.")
        print(f"Number of topics generated: {len(set(topics))}")
        print(f"Training time: {end_time - self.start_time:.2f} seconds.")

        # Save the BERTopic model using safetensors
        try:
            print("Saving BERTopic model...")
            embedding_model = self.config.get("finbert_model_path", "all-MiniLM-L12-v2")
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
        """
        Train BERTopic model in an iterative manner.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        None
        """
        print("Initializing iterative BERTopic model...")

        # Split the input documents into chunks
        doc_chunks = [docs[i:i+self.doc_chunk_size] for i in range(0, len(docs), self.doc_chunk_size)]

        if not doc_chunks:
            print("No document chunks found for iterative training.")
            return

        # Initialize the base model without embeddings
        self.topic_model = self._initialize_bertopic_model()

        # Start timer
        self.start_time = time.time()

        # Initialize the stop event for the heartbeat thread
        stop_event = threading.Event()

        # Start the heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat, args=(stop_event,))
        heartbeat_thread.daemon = True  # Ensures the thread exits when the main program does
        heartbeat_thread.start()

        # Iterate over the chunks and update the model
        for idx, chunk in enumerate(doc_chunks):
            print(f"Processing chunk {idx+1}/{len(doc_chunks)}...")
            try:
                # Compute embeddings for the current chunk
                embeddings = self.compute_embeddings(chunk)

                # Fit the model on the current chunk
                self.topic_model.partial_fit(chunk, embeddings=embeddings)

                print(f"Chunk {idx+1} processed.")

            except Exception as e:
                print(f"An error occurred during iterative training: {e}")
                # Stop the heartbeat thread in case of an error
                stop_event.set()
                heartbeat_thread.join()
                return

        # Stop the heartbeat thread after training completes
        stop_event.set()
        heartbeat_thread.join()

        # End timer
        end_time = time.time()

        # Retrieve topics
        topics = self.topic_model.get_topics()

        # Print information about the training process
        print(f"Iterative BERTopic model trained on {len(docs)} documents.")
        print(f"Number of topics generated: {len(topics)}")
        print(f"Training time: {end_time - self.start_time:.2f} seconds.")

        # Save the final merged model
        try:
            print("Saving the final merged BERTopic model using safetensors...")
            embedding_model = self.config.get("finbert_model_path", "all-MiniLM-L12-v2")
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

    This function loads the configuration from `config.json`, sets the random seed, and extracts the necessary variables from the config.
    Then, it initializes the `FileHandler` and `TextProcessor` classes with the imported configuration, creates the ECC sample, and extracts the relevant sections.
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
