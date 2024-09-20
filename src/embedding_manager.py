# embedding_manager.py

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import multiprocessing
from multiprocessing import Pool
import GPUtil
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs will be printed to stdout
    ]
)

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
    np.ndarray or None
        The computed embeddings or None if an error occurs.
    """
    try:
        # Set the CUDA device
        if device_id != -1:
            torch.cuda.set_device(device_id)
            logging.info(f"Worker {multiprocessing.current_process().name} assigned to GPU {device_id}")
            logging.debug(f"Current device: {torch.cuda.current_device()}, Device count: {torch.cuda.device_count()}")
        else:
            logging.info(f"Worker {multiprocessing.current_process().name} using CPU")

        # Initialize the embedding model based on the choice
        if embedding_choice == "all-MiniLM-L12-v2":
            model_name = "sentence-transformers/all-MiniLM-L12-v2"
            logging.info(f"Loading embedding model: {model_name} on {'CPU' if device_id == -1 else f'GPU {device_id}'}")
            model = SentenceTransformer(model_name, device=f'cuda:{device_id}' if device_id != -1 else 'cpu')
        elif embedding_choice == "finbert-local":
            if not os.path.exists(finbert_model_path):
                raise ValueError(f"The specified model path does not exist: {finbert_model_path}")
            model_name = finbert_model_path
            logging.info(f"Loading embedding model: {model_name} on {'CPU' if device_id == -1 else f'GPU {device_id}'}")
            model = SentenceTransformer(model_name, device=f'cuda:{device_id}' if device_id != -1 else 'cpu')
        elif embedding_choice == "finbert-pretrain":
            model_name = "yiyanghkust/finbert-pretrain"
            logging.info(f"Loading embedding pipeline: {model_name} on {'CPU' if device_id == -1 else f'GPU {device_id}'}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.model_max_length = 512
            tokenizer.truncation = True

            # Initialize the pipeline on the specified device
            pipe = pipeline(
                "feature-extraction",
                model=model_name,
                tokenizer=tokenizer,
                device=device_id if device_id != -1 else -1  # CUDA device ID or CPU
            )

            # Compute embeddings using the pipeline
            embeddings = []
            for doc in docs:
                features = pipe(doc)
                # Flatten the list of lists and convert to np.ndarray
                flat_features = np.array(features).flatten()
                embeddings.append(flat_features)

            # Clean up resources
            del pipe
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            logging.info(f"Completed embedding for {len(docs)} documents using {model_name} on {'CPU' if device_id == -1 else f'GPU {device_id}'}")
            return np.array(embeddings)
        else:
            raise ValueError(f"Unknown embedding model choice: {embedding_choice}")

        # Compute embeddings using the SentenceTransformer model
        embeddings = model.encode(docs, batch_size=32, show_progress_bar=False)
        logging.info(f"Completed embedding for {len(docs)} documents using {model_name} on {'CPU' if device_id == -1 else f'GPU {device_id}'}")

        # Clean up resources
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return embeddings

    except Exception as e:
        logging.error(f"Error in compute_embeddings_worker on device {device_id}: {e}")
        return None

class EmbeddingManager:
    """
    A class to manage embedding computations across multiple GPUs or CPU.
    """

    def __init__(self, embedding_choice="all-MiniLM-L12-v2", finbert_model_path="", batch_size=32):
        """
        Initialize the EmbeddingManager.

        Parameters
        ----------
        embedding_choice : str, optional
            The choice of embedding model. Default is "all-MiniLM-L12-v2".
        finbert_model_path : str, optional
            Path to the FinBERT model if using a local model. Default is "".
        batch_size : int, optional
            Batch size for embedding computation. Default is 32.
        """
        self.embedding_choice = embedding_choice
        self.finbert_model_path = finbert_model_path
        self.batch_size = batch_size

        # Detect number of GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logging.info(f"Multiple GPUs detected: {self.num_gpus} GPU(s) will be used.")
            self.devices = list(range(self.num_gpus))  # GPU IDs: 0, 1, ..., num_gpus-1
        elif self.num_gpus == 1:
            logging.info("Single GPU detected. Using GPU 0.")
            self.devices = [0]
        else:
            logging.info("No GPU detected. Using CPU.")
            self.devices = [-1]  # -1 indicates CPU

        # Log the embedding model choice
        if self.embedding_choice == "finbert-local":
            logging.info(f"Using local FinBERT model at: {self.finbert_model_path}")
        elif self.embedding_choice == "finbert-pretrain":
            logging.info("Using pre-trained FinBERT model: yiyanghkust/finbert-pretrain")
        else:
            logging.info(f"Using embedding model: {self.embedding_choice}")

    def compute_embeddings(self, docs):
        """
        Compute embeddings for the given documents using available GPUs or CPU.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        np.ndarray or None
            The computed embeddings or None if an error occurs.
        """
        if self.devices == [-1]:
            # CPU mode
            logging.info("Computing embeddings on CPU...")
            if self.embedding_choice == "finbert-pretrain":
                try:
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

                    # Convert to numpy array
                    embeddings = np.array(embeddings)

                    # Clean up resources
                    del pipe
                    del tokenizer
                    torch.cuda.empty_cache()
                    gc.collect()
                    logging.info(f"Completed embedding on CPU for {len(docs)} documents.")
                    return embeddings

                except Exception as e:
                    logging.error(f"Error during CPU embedding computation: {e}")
                    return None

            else:
                try:
                    model_name = self.embedding_choice if self.embedding_choice != "finbert-local" else self.finbert_model_path
                    logging.info(f"Loading embedding model for CPU: {model_name}")
                    model = SentenceTransformer(
                        model_name,
                        device='cpu'
                    )
                    embeddings = model.encode(docs, batch_size=self.batch_size, show_progress_bar=True)
                    logging.info(f"Completed embedding on CPU for {len(docs)} documents.")

                    # Clean up resources
                    del model
                    gc.collect()
                    return embeddings

                except Exception as e:
                    logging.error(f"Error during CPU embedding computation with SentenceTransformer: {e}")
                    return None

        else:
            # GPU mode
            logging.info(f"Computing embeddings on {len(self.devices)} GPU(s)...")
            try:
                # Split docs into chunks based on the number of GPUs
                chunks = np.array_split(docs, len(self.devices))

                # Prepare arguments for each worker
                args = []
                for i, chunk in enumerate(chunks):
                    args.append((chunk.tolist(), self.embedding_choice, self.finbert_model_path, self.devices[i]))

                # Use multiprocessing Pool with 'spawn' start method
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=len(self.devices)) as pool:
                    results = pool.starmap(compute_embeddings_worker, args)

                # Filter out any None results due to errors
                results = [r for r in results if r is not None]
                if not results:
                    logging.error("No embeddings were successfully computed.")
                    return None

                # Concatenate all embeddings
                embeddings = np.vstack(results)
                logging.info(f"Completed embedding on GPUs for {len(docs)} documents.")

                # Clean up resources
                del results
                torch.cuda.empty_cache()
                gc.collect()
                return embeddings

            except Exception as e:
                logging.error(f"Error during GPU embedding computation: {e}")
                return None

    def get_gpu_status(self):
        """
        Get the current status of GPUs.

        Returns
        -------
        list
            A list of strings describing the status of each GPU.
        """
        if self.devices == [-1]:
            return ["CPU"]
        else:
            gpus = GPUtil.getGPUs()
            status = []
            for device_id in self.devices:
                gpu = next((gpu for gpu in gpus if gpu.id == device_id), None)
                if gpu:
                    status.append(
                        f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}MB/{gpu.memoryTotal}MB memory"
                    )
                else:
                    status.append(f"GPU {device_id}: Not Available")
            return status
