import os
import json
import time
import numpy as np
import torch  # For checking if GPU is available
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
            A dictionary containing the configuration for the model. The dictionary should contain the following keys:
                - model_save_path: str
                    The path to save the trained model.
                - modeling_type: str
                    The type of modeling to use. Options are "regular", "iterative", "iterative_zeroshot", or "zeroshot".
                - doc_chunk_size: int
                    The number of documents to process in each chunk. Only used for iterative modeling.

        Attributes
        ----------
        config : dict
            The configuration dictionary.
        model_save_path : str
            The path to save the trained model.
        modeling_type : str
            The type of modeling used.
        doc_chunk_size : int
            The number of documents to process in each chunk. (only for iterative modeling)
        device : str
            The device to use for training, either "cpu" or "cuda" if gpu is available.
        topic_model : BERTopic
            The trained topic model.
        model : SentenceTransformer
            The trained sentence transformer model.
        """
        self.config = config
        self.model_save_path = config["model_save_path"]
        self.modeling_type = config.get("modeling_type", "regular")  # default, options: ["regular", "iterative", "iterative_zeroshot", "zeroshot"]
        self.doc_chunk_size = config.get("doc_chunk_size", 5000)  # default, this variable is only used for iterative training
        self.device = self._select_device()  # Add device selection here
        self.topic_model = None
        self.model = self._select_embedding_model(config)  # Load the appropriate embedding model based on config

    def _select_device(self):
        # Check if GPU is available and return the correct device
        """Check if GPU is available and return the correct device.

        Returns
        -------
        device : torch.device
            The device to use for training, either "cpu" or "cuda" if gpu is available.
        """
        if torch.cuda.is_available():
            print("GPU is available. Using GPU...")
            return torch.device("cuda")
        else:
            print("GPU not available. Falling back to CPU...")
            return torch.device("cpu")

    def _select_embedding_model(self, config):
        # Select the embedding model based on the config setting
        """
        Select the embedding model based on the config setting. Possible choices are:

        - "all-MiniLM-L12-v2": SentenceTransformer model: all-MiniLM-L12-v2
        - "finbert-local": FinBERT model from local path
        - "finbert-pretrain": FinBERT model from HuggingFace pipeline

        Parameters
        ----------
        config : dict
            The configuration dictionary containing the embedding model choice.

        Returns
        -------
        model : SentenceTransformer
            The selected embedding model.

        Raises
        ------
        ValueError
            If the specified embedding model choice is unknown.
        """

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
        """
        Load the FinBERT model from the Hugging Face pipeline.

        Parameters
        ----------
        None

        Returns
        -------
        pipe : Pipeline
            The loaded FinBERT model pipeline.

        Notes
        -----
        This function returns a Hugging Face pipeline object with the FinBERT model and tokenizer.
        The `device` parameter is set to 0 if the device is "cuda", otherwise it is set to -1.
        """
        
        print(f"Loading FinBERT model pipeline from HuggingFace on {self.device}...")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
        
        # Force tokenizer to truncate inputs at 512 tokens
        tokenizer.model_max_length = 512
        tokenizer.truncation = True

        # Set up the pipeline with the model and tokenizer
        pipe = pipeline("feature-extraction",
                        model="yiyanghkust/finbert-pretrain",
                        tokenizer=tokenizer,
                        device=0 if self.device == "cuda" else -1)
        return pipe


    def _initialize_bertopic_model(self):
        """
        Initialize the BERTopic model with the specified parameters.

        Parameters
        ----------
        None

        Returns
        -------
        BERTopic
            The initialized BERTopic model.
        """
        print(f"Embedding Model used: {self.model}...")
        # Initialize CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=tuple(self.config["vectorizer_model_params"]["ngram_range"]),
                                           stop_words=self.config["vectorizer_model_params"]["stop_words"],
                                           min_df=self.config["vectorizer_model_params"]["min_df"])

        # Initialize UMAP
        umap_model = UMAP(n_neighbors=self.config["umap_model_params"]["n_neighbors"],
                          n_components=self.config["umap_model_params"]["n_components"],
                          min_dist=self.config["umap_model_params"]["min_dist"],
                          metric=self.config["umap_model_params"]["metric"],
                          random_state=42)

        # Initialize HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=self.config["hdbscan_model_params"]["min_cluster_size"],
                                metric=self.config["hdbscan_model_params"]["metric"],
                                cluster_selection_method=self.config["hdbscan_model_params"]["cluster_selection_method"],
                                prediction_data=self.config["hdbscan_model_params"]["prediction_data"])

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
                representation_model=[keybert_model, mmr_model]
            )
        else:
            print("Initializing regular BERTopic model...")
            return BERTopic(
                embedding_model=self.model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=[keybert_model, mmr_model]  # Combined representation
            )

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
        start_time = time.time()

        # Train the BERTopic model
        print(f"Training BERTopic model using the following modeling type: {self.modeling_type}...")
        try:
            topics, probs = self.topic_model.fit_transform(docs)

            # Print document count per topic for debugging
            topic_doc_count = {topic: topics.count(topic) for topic in set(topics)}
            print("Document count per topic:", topic_doc_count)

        except Exception as e:
            print(f"An error occurred during ctf-idf transformation: {e}")
            return

        # End timer
        end_time = time.time()

        # Print information about the training process
        print(f"BERTopic model trained on {len(docs)} sections.")
        print(f"Number of topics generated: {len(set(topics))}")
        print(f"Training time: {end_time - start_time:.2f} seconds.")

        # Save the BERTopic model using safetensors
        try:
            print("Saving BERTopic model...")
            embedding_model = self.config["finbert_model_path"]
            self.topic_model.save(self.model_save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
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

        # Initialize the base model with the first chunk of documents
        base_model = self._initialize_bertopic_model().fit(doc_chunks[0])

        # Iterate over the remaining chunks of documents
        for chunk in doc_chunks[1:]:
            print("Merging new documents into the base model...")

            # Train a new model on the current chunk of documents
            new_model = self._initialize_bertopic_model().fit(chunk)

            # Merge the new model with the base model
            updated_model = BERTopic.merge_models([base_model, new_model])

            # Print the number of newly discovered topics
            nr_new_topics = len(set(updated_model.topics_)) - len(set(base_model.topics_))
            new_topics = list(updated_model.topic_labels_.values())[-nr_new_topics:]
            print("The following topics are newly found:")
            print(f"{new_topics}\n")

            # Update the base model
            base_model = updated_model

        # Save the final merged model
        self.topic_model = base_model

        print("Saving the final merged BERTopic model using safetensors...")
        embedding_model = self.config["finbert_model_path"]
        self.topic_model.save(self.model_save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
        print(f"Final BERTopic model saved to {self.model_save_path}.")


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
