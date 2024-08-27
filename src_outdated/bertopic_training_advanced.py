import os
import time
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from file_handling import read_index_file, create_ecc_sample

# Define variables at the top of the script
index_file_path = "D:/daten_masterarbeit/list_earnings_call_transcripts.csv"
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
sample_size = 10  # Number of companies to train on
document_split = "paragraphs"
section_to_analyze = "Presentation"
random_seed = 42
model_save_path = "D:/daten_masterarbeit/bertopic_model.pkl"  # Include filename

# BERTopic parameters to adjust the number of topics
n_gram_range = (1, 2)  # Use bigrams to capture more context
min_topic_size = 100  # Set the minimum topic size to reduce the number of small topics
nr_topics = 50  # Set the maximum number of topics

# Load FinBERT model using SentenceTransformer
embedding_model = SentenceTransformer("yiyanghkust/finbert-tone")

# Define UMAP model to improve topic separation
umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.2, metric='cosine')

# Define HDBSCAN model for better clustering
hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Define CountVectorizer with custom stopwords and modified settings
stopwords = set(['operator', 'another_common_term'])  # Add more terms as needed
vectorizer_model = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords, max_df=0.8, min_df=2)

# Initialize BERTopic with adjusted parameters
topic_model = BERTopic(
    representation_model=KeyBERTInspired(),
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=min_topic_size,
    nr_topics=nr_topics
)

def split_document(company, call_id, company_info, date, text, section_to_analyze, document_split):
    from text_splitting import extract_and_split_section  # Import here to avoid circular imports
    return extract_and_split_section(company, call_id, company_info, date, text, document_split, section_to_analyze)

def train_bertopic_model_on_companies(index_file_path, folderpath_ecc, sample_size, document_split, section_to_analyze, random_seed, model_save_path):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Read the index file
    index_file = read_index_file(index_file_path)
    print("Index file loaded successfully.")

    # Create sample of earnings conference calls
    ecc_sample = create_ecc_sample(sample_size, index_file, folderpath_ecc)
    print(f"Sample of {sample_size} companies created for training.")

    all_relevant_sections = []

    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            company_info, date, text = value
            relevant_section = split_document(permco, call_id, company_info, date, text, section_to_analyze, document_split)
            if relevant_section is not None:
                if isinstance(relevant_section, list):
                    all_relevant_sections.extend(relevant_section)
                else:
                    all_relevant_sections.append(relevant_section)

    # Ensure all elements in all_relevant_sections are strings
    all_relevant_sections = [str(section) for section in all_relevant_sections]

    # Fit the BERTopic model once on all the relevant sections
    if all_relevant_sections:
        print("Fitting BERTopic...")
        start_time = time.time()  # Start timing the training process
        topics, probabilities = topic_model.fit_transform(all_relevant_sections)
        end_time = time.time()  # End timing the training process
        training_time = end_time - start_time
        print(f"BERTopic model trained on {len(all_relevant_sections)} sections.")
        print(f"Number of topics generated: {len(set(topics))}")
        print(f"Training time: {training_time:.2f} seconds")

        # Save the trained model
        topic_model.save(model_save_path)
        print(f"BERTopic model saved to {model_save_path}")
    else:
        print("No relevant sections found to fit BERTopic.")

def load_bertopic_model(model_load_path):
    # Load the trained BERTopic model
    topic_model = BERTopic.load(model_load_path)
    print(f"BERTopic model loaded from {model_load_path}")
    return topic_model

# Execute the training function with defined variables
if __name__ == "__main__":
    train_bertopic_model_on_companies(
        index_file_path=index_file_path,
        folderpath_ecc=folderpath_ecc,
        sample_size=sample_size,
        document_split=document_split,
        section_to_analyze=section_to_analyze,
        random_seed=random_seed,
        model_save_path=model_save_path
    )
