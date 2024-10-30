# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:40:42 2024

@author: nikla
"""

# Step 1: Import necessary modules and classes
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import sys
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import logging

# Import PCA
from sklearn.decomposition import PCA

# Enable logging for Gensim to monitor progress
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # To override any previous configurations

# Download necessary NLTK data files (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Check if "src" is not part of the path
if "src" not in current_dir:
    # Adjust path to include 'src'
    src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
    sys.path.append(src_path)

from file_handling import FileHandler
from text_processing import TextProcessor

# Step 2: Load configuration
config_file_path = r'C:\Users\nikla\OneDrive\Dokumente\winfoMaster\Masterarbeit\bertopic_ecc\config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Step 3: Extract variables from configuration
random_seed = config["random_seed"]
np.random.seed(random_seed)  # Set the random seed for reproducibility
index_file_ecc_folder = config["index_file_ecc_folder"]
folderpath_ecc = config["folderpath_ecc"]
sample_size = config["sample_size"]
document_split = config["document_split"]
section_to_analyze = config["section_to_analyze"]
model_load_path = config["model_load_path"]
embedding_model_choice = config["embedding_model_choice"]
modeling_type = config.get("modeling_type", "regular")  # Get modeling type from config

# Add a boolean to skip manual labeling
skip_manual_labeling = True  # Set to True to skip manual labeling

# Initialize FileHandler and TextProcessor with the imported configuration
print("Initializing file handler and text processor...")
file_handler = FileHandler(config=config)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

# Read the index file
print("Reading index file...")
index_file = file_handler.read_index_file()

# Create ECC sample
print("Creating ECC sample...")
ecc_sample = file_handler.create_ecc_sample(sample_size)

# Extract texts for BERTopic analysis (processed sections/paragraphs)
print("Extracting and processing relevant sections...")
all_relevant_sections = []
dates = []  # To store dates for temporal analysis
for permco, calls in ecc_sample.items():
    for call_id, value in calls.items():
        company_info = value['company_name']
        date = value['date']  # Assuming date is available in the desired format
        text = value['text_content']
        relevant_sections = text_processor.extract_and_split_section(permco, call_id, company_info, date, text)
        all_relevant_sections.extend(relevant_sections)
        # Add the relevant sections to the ECC sample
        value['relevant_sections'] = relevant_sections
        # Append date for each relevant section (assuming each section corresponds to the call date)
        dates.extend([date] * len(relevant_sections))

# Load the BERTopic model
print("Loading BERTopic model...")
if embedding_model_choice.lower() == 'default':
    topic_model = BERTopic.load(model_load_path)
else:
    # Try to use CUDA if available, else fallback to CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embedding_model = SentenceTransformer(embedding_model_choice, device=device)
    topic_model = BERTopic.load(model_load_path, embedding_model=embedding_model)

# Compute embeddings for all documents
print("Computing embeddings for all documents...")
try:
    embeddings = topic_model.embedding_model.embed_documents(all_relevant_sections)
except AttributeError:
    # If 'embed_documents' method is not available, use 'encode' method
    embeddings = topic_model.embedding_model.encode(all_relevant_sections, show_progress_bar=True)

# Apply PCA if modeling type is 'regular'
if modeling_type.lower() == 'regular':
    print("Applying PCA to embeddings...")
    n_components = 50  # Adjust the number of components as needed
    pca = PCA(n_components=n_components, random_state=random_seed)
    embeddings = pca.fit_transform(embeddings)
    print(f"Embeddings reduced to {n_components} dimensions using PCA.")
else:
    print("Skipping PCA as modeling type is 'zeroshot'.")

# Transform the documents to get topics and probabilities using embeddings
print("Transforming documents...")
topics, probs = topic_model.transform(all_relevant_sections, embeddings=embeddings)

# Create a DataFrame for evaluation
print("Creating evaluation DataFrame...")
df_eval = pd.DataFrame({
    'Document': all_relevant_sections,
    'Topic': topics,
    'Probability': probs,
    'Date': dates  # Include dates for temporal analysis
})

# Convert 'Date' column to datetime
df_eval['Date'] = pd.to_datetime(df_eval['Date'], errors='coerce')

# 1. Identify the Top N Most Frequent Topics
print("Identifying top most frequent topics...")
topic_counts = df_eval['Topic'].value_counts()
topic_counts = topic_counts[topic_counts.index != -1]  # Exclude outlier topic
top_n_topics = 5  # Adjust this number as needed
top_topics = topic_counts.nlargest(top_n_topics).index.tolist()
print("Top Topics:")
print(top_topics)

# Save topic frequencies to a txt file
with open('topic_frequencies.txt', 'w') as f:
    f.write("Topic Frequencies (excluding outlier topic -1):\n")
    f.write(topic_counts.to_string())
print("Topic frequencies saved to 'topic_frequencies.txt'.")

# Tokenize documents (since they are preprocessed, tokenization is sufficient)
print("Tokenizing documents...")
tokenized_texts = [word_tokenize(doc.lower()) for doc in all_relevant_sections]

# Limit the dataset size for testing (e.g., first 100 documents)
max_docs = 100  # Adjust as needed
tokenized_texts = tokenized_texts[:max_docs]
df_eval = df_eval.iloc[:max_docs]

# Create Gensim dictionary and corpus
print("Creating Gensim dictionary and corpus...")
dictionary = Dictionary(tokenized_texts)

# Filter out rare tokens to reduce vocabulary size
print("Filtering out rare tokens...")
dictionary.filter_extremes(no_below=2)  # Keep tokens that appear in at least 2 documents

# Update tokenized_texts to remove words not in the dictionary
print("Updating tokenized texts to include only tokens in the dictionary...")
tokenized_texts = [[token for token in text if token in dictionary.token2id] for text in tokenized_texts]

# Create corpus
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# Get top words for top topics
print("Getting top words for top topics...")
topics_info = topic_model.get_topics()

# Adjust the number of top words per topic and ensure words are in the dictionary
top_k = 2  # Use only top 2 words per topic
top_words_per_topic = []
valid_topic_ids = []
for topic_id in top_topics:
    words = []
    for word, _ in topics_info[topic_id]:
        word_lower = word.lower()  # Ensure consistent casing
        if word_lower in dictionary.token2id:
            words.append(word_lower)
        if len(words) == top_k:
            break
    if words:
        top_words_per_topic.append(words)
        valid_topic_ids.append(topic_id)
    else:
        print(f"Warning: Topic {topic_id} has no words in the dictionary after filtering and will be skipped.")

# Check if any topics have empty word lists
if not top_words_per_topic:
    print("Error: No valid topics with words found. Cannot compute coherence.")
else:
    # Proceed to compute coherence
    # Use U_Mass coherence measure
    num_processes = 1  # Disable multiprocessing
    print("Calculating topic coherence using U_Mass measure...")
    coherence_model_umass = CoherenceModel(
        topics=top_words_per_topic,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass',
        processes=num_processes
    )

    # Time the coherence calculation
    import time
    start_time = time.time()
    coherence_scores_umass = coherence_model_umass.get_coherence_per_topic()
    end_time = time.time()
    print(f"Coherence calculation took {end_time - start_time:.2f} seconds.")

    # Print U_Mass coherence scores
    print("\nTopic Coherence (U_Mass) Scores:")
    for idx, topic_id in enumerate(valid_topic_ids):
        score = coherence_scores_umass[idx]
        print(f"Topic {topic_id}: U_Mass Coherence Score = {score:.4f}")

    # Save coherence scores to a txt file
    with open('coherence_scores.txt', 'w') as f:
        f.write("Topic Coherence (U_Mass) Scores:\n")
        for idx, topic_id in enumerate(valid_topic_ids):
            score = coherence_scores_umass[idx]
            f.write(f"Topic {topic_id}: U_Mass Coherence Score = {score:.4f}\n")
    print("Coherence scores saved to 'coherence_scores.txt'.")

    # Calculate average U_Mass coherence
    average_umass = np.mean(coherence_scores_umass)
    print(f"\nAverage U_Mass Coherence (Top Topics): {average_umass:.4f}")

# Proceed with manual inspection and other analyses
# Analyze topic distribution for valid topics
print("Analyzing topic distribution for valid topics...")
df_top_topics = df_eval[df_eval['Topic'].isin(valid_topic_ids)]
topic_counts_top = df_top_topics['Topic'].value_counts().sort_index()

# Save topic distribution to a txt file
with open('topic_distribution.txt', 'w') as f:
    f.write("Topic Distribution of Valid Topics:\n")
    f.write(topic_counts_top.to_string())
print("Topic distribution saved to 'topic_distribution.txt'.")

# Plot topic distribution
plt.figure(figsize=(8, 6))
topic_counts_top.plot(kind='bar')
plt.xlabel('Topic')
plt.ylabel('Number of Documents')
plt.title('Topic Distribution of Valid Topics')
plt.savefig('topic_distribution.png')
plt.close()
print("Topic distribution plot saved to 'topic_distribution.png'.")

# Analyze average probability per topic for valid topics
print("Analyzing average probability per topic for valid topics...")
avg_prob_per_topic = df_top_topics.groupby('Topic')['Probability'].mean()
for topic_id, avg_prob in avg_prob_per_topic.items():
    print(f"Topic {topic_id}: Average Probability = {avg_prob:.4f}")

# Save average probabilities to a txt file
with open('average_probabilities.txt', 'w') as f:
    f.write("Average Probability per Topic:\n")
    for topic_id, avg_prob in avg_prob_per_topic.items():
        f.write(f"Topic {topic_id}: Average Probability = {avg_prob:.4f}\n")
print("Average probabilities saved to 'average_probabilities.txt'.")

# Manual inspection and labeling of valid topics
if not skip_manual_labeling:
    print("Manual inspection and labeling of valid topics...")
    topic_info = topic_model.get_topic_info()
    topic_info_top = topic_info[topic_info['Topic'].isin(valid_topic_ids)]

    # Create a dictionary to store topic labels
    topic_labels = {}

    for index, row in topic_info_top.iterrows():
        topic_id = row['Topic']
        top_words = [word for word, _ in topic_model.get_topic(topic_id)[:top_k]]
        print(f"\nTopic {topic_id}")
        print(f"Top Words: {', '.join(top_words)}")

        # Get sample documents
        sample_docs = df_eval[df_eval['Topic'] == topic_id]['Document'].head(3)
        print("Sample Documents:")
        for idx, doc in enumerate(sample_docs, 1):
            print(f"\nSample Document {idx}:\n")
            print(doc)
            print("\n" + "-" * 80)

        # Assign labels based on manual inspection
        label = input(f"\nPlease assign a label for Topic {topic_id}: ")
        topic_labels[topic_id] = label

    # Add labels to topic_info_top DataFrame
    topic_info_top['Label'] = topic_info_top['Topic'].map(topic_labels)

    # Display the labeled topics
    print("\nLabeled Topics:")
    print(topic_info_top[['Topic', 'Label']])

    # Save labeled topics to a txt file
    with open('labeled_topics.txt', 'w') as f:
        f.write("Labeled Topics:\n")
        f.write(topic_info_top[['Topic', 'Label']].to_string(index=False))
    print("Labeled topics saved to 'labeled_topics.txt'.")
else:
    print("Skipping manual labeling as per user request.")

# Calculate topic diversity for valid topics
print("Calculating topic diversity for valid topics...")

def calculate_topic_diversity(top_words_per_topic):
    unique_words = set()
    total_words = 0
    for words in top_words_per_topic:
        unique_words.update(words)
        total_words += len(words)
    diversity = len(unique_words) / total_words if total_words > 0 else 0
    return diversity

topic_diversity = calculate_topic_diversity(top_words_per_topic)
print(f"\nTopic Diversity (Valid Topics): {topic_diversity:.4f}")

# Save topic diversity to a txt file
with open('topic_diversity.txt', 'w') as f:
    f.write(f"Topic Diversity (Valid Topics): {topic_diversity:.4f}")
print("Topic diversity saved to 'topic_diversity.txt'.")

# 3. Extract Documents Assigned to Outlier Topic (-1)
print("Extracting documents assigned to outlier topic (-1)...")
outlier_docs = df_eval[df_eval['Topic'] == -1]['Document']
outlier_docs_list = outlier_docs.tolist()
print(f"\nNumber of documents assigned to outlier topic (-1): {len(outlier_docs_list)}")

# Save outlier documents to a text file
with open('outlier_documents.txt', 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(outlier_docs_list):
        f.write(f"Document {idx+1}:\n{doc}\n\n{'-'*80}\n\n")
print("Outlier documents saved to 'outlier_documents.txt'.")

# Optional: Print first few outlier documents
print("\nFirst 5 Outlier Documents:")
for idx, doc in enumerate(outlier_docs_list[:5]):
    print(f"\nOutlier Document {idx+1}:\n{doc}\n")
    print("---")

# Save final remarks to a txt file
with open('final_remarks.txt', 'w') as f:
    f.write("Evaluation complete. Please review the outputs and results.\n")
print("\nEvaluation complete. Please review the outputs and results.")
print("Final remarks saved to 'final_remarks.txt'.")
