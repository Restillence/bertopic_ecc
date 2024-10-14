# utils.py
import json
from bertopic import BERTopic
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import time

def print_configuration(config):
    """
    Print the configuration dictionary.

    Parameters:
    - config (dict): The configuration dictionary to be printed.
    """
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def load_bertopic_model(model_path):
    """
    Load the trained BERTopic model from a file.

    Parameters:
    - model_path (str): The path to the saved BERTopic model file.

    Returns:
    - BERTopic: The loaded BERTopic model.
    """
    topic_model = BERTopic.load(model_path)
    print(f"BERTopic model loaded from {model_path}")
    return topic_model

def heartbeat():
    """
    Prints a heartbeat message to the console every 5 minutes.
    Runs indefinitely until the main program exits.
    """
    while True:
        time.sleep(300)  # 300 seconds = 5 minutes
        print("[Heartbeat] The script is still running...")

def process_topics(path, output_path, topics_to_keep):
    # Load the CSV file
    df = pd.read_csv(path)

    # Function to keep only the specified topics and corresponding texts
    def keep_topics_and_texts(row, topics_to_keep):
        topics = eval(row['topics'])
        texts = eval(row['text'])
        filtered_data = [(topic, text) for topic, text in zip(topics, texts) if topic in topics_to_keep]
        if filtered_data:
            filtered_topics, filtered_texts = zip(*filtered_data)
        else:
            filtered_topics, filtered_texts = [], []
        return list(filtered_topics), list(filtered_texts)

    # Apply the function to each row
    df[['filtered_topics', 'filtered_texts']] = df.apply(lambda row: keep_topics_and_texts(row, topics_to_keep), axis=1, result_type='expand')

    # Consistency check to validate if topics and texts are of the same length
    def check_consistency(row):
        topics = eval(row['topics'])
        texts = eval(row['text'])
        return len(topics) == len(texts)

    df['consistent'] = df.apply(check_consistency, axis=1)

    # Export to CSV; not neccessary for the final version
    #df.to_csv(output_path, sep='\t', encoding='utf-8', index=False, header=list(df))

    return df[['topics', 'text', 'filtered_topics', 'filtered_texts', 'consistent', 'call_id', 'permco', 'date']]

def create_transition_matrix(topic_sequence, num_topics):
    transition_matrix = np.zeros((num_topics, num_topics))
    for i in range(len(topic_sequence) - 1):
        from_topic = topic_sequence[i]
        to_topic = topic_sequence[i + 1]
        transition_matrix[from_topic][to_topic] += 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    return transition_matrix

def compute_similarity_to_average(df, num_topics):
    # Create transition matrices
    transition_matrices = []
    call_ids = []
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        transition_matrix = create_transition_matrix(topics, num_topics)
        transition_matrices.append(transition_matrix)
        call_ids.append(call_id)

    # Compute the average transition matrix
    average_transition_matrix = np.mean(transition_matrices, axis=0)
    average_transition_matrix = np.nan_to_num(average_transition_matrix)

    # Compute similarity between each call's transition matrix and the average
    similarities = []
    for tm in transition_matrices:
        tm = np.nan_to_num(tm)
        tm_vector = tm.flatten()
        avg_vector = average_transition_matrix.flatten()
        sim = 1 - cosine(tm_vector, avg_vector)
        similarities.append(sim)

    # Create a DataFrame with call_ids and similarities
    similarity_df = pd.DataFrame({'call_id': call_ids, 'similarity_to_average': similarities})

    return similarity_df
