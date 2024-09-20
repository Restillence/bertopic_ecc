# embedding_manager.py

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import multiprocessing
from multiprocessing import Pool
import GPUtil


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
            print(f"Multiple GPUs detected: {self.num_gpus} GPU(s) will be used.")
            self.devices = list(range(self.num_gpus))  # GPU IDs: 0, 1, ..., num_gpus-1
        elif self.num_gpus == 1:
            print("Single GPU detected. Using GPU 0.")
            self.devices = [0]
        else:
            print("No GPU detected. Using CPU.")
            self.devices = [-1]  # -1 indicates CPU

    def compute_embeddings(self, docs):
        """
        Compute embeddings for the given documents using available GPUs or CPU.

        Parameters
        ----------
        docs : list
            A list of strings representing the input documents.

        Returns
        -------
        np.ndarray
            The computed embeddings.
        """
        if self.devices == [-1]:
            # CPU mode
            print("Computing embeddings on CPU...")
            if self.embedding_choice == "finbert-pretrain":
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
                model = SentenceTransformer(
                    self.embedding_choice if self.embedding_choice != "finbert-local" else self.finbert_model_path,
                    device='cpu'
                )
                embeddings = model.encode(docs, batch_size=self.batch_size, show_progress_bar=True)
                return embeddings
        else:
            # GPU mode
            print(f"Computing embeddings on {len(self.devices)} GPU(s)...")
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

            # Concatenate all embeddings
            embeddings = np.vstack(results)
            return embeddings

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
