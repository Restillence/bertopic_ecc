# utils.py
import json
from bertopic import BERTopic
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import time
import ast

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

def process_topics(path, output_path, topics_to_keep, threshold_percentage=None):
    # Load the CSV file
    df = pd.read_csv(path)
    if topics_to_keep == "all":
        df['filtered_topics'] = df["topics"]
        df['filtered_texts'] = df["text"]
        return df
    # Convert string representations of lists into actual lists using ast.literal_eval
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['text'] = df['text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Automatically determine topics to keep if set to "auto"
    if topics_to_keep == "auto":
        topics_to_keep = determine_topics_to_keep(df, threshold_percentage)

    # Function to keep only the specified topics and corresponding texts
    def keep_topics_and_texts(row, topics_to_keep):
        topics = row['topics']
        texts = row['text']
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
        return len(row['topics']) == len(row['text'])

    df['consistent'] = df.apply(check_consistency, axis=1)

    # Return relevant columns
    return df[['topics', 'text', 'filtered_topics', 'filtered_texts', 'consistent', 'call_id', 'permco', 'date']]

def determine_topics_to_keep(df, threshold_percentage):
    """
    Determine topics that appear in at least 1 call for the specified percentage of companies.
    """
    # Get the total number of companies
    total_companies = df['permco'].nunique()

    # Set to keep track of topics per company
    topics_per_company = df.groupby('permco')['topics'].apply(lambda x: set().union(*x)).reset_index()

    # Count the occurrences of each topic across companies
    topic_counts = {}

    for topics in topics_per_company['topics']:
        for topic in topics:
            if topic not in topic_counts:
                topic_counts[topic] = 0
            topic_counts[topic] += 1

    # Determine the threshold number of companies
    company_threshold = total_companies * (threshold_percentage / 100)

    # Find topics that appear in at least the threshold percentage of companies
    topics_to_keep = {topic for topic, count in topic_counts.items() if count >= company_threshold}
    
    # Find topics to remove
    topics_to_remove = set(topic_counts.keys()) - topics_to_keep

    # Print statements to show kept and removed topics
    print(f"Topics to Keep (Appearing in {threshold_percentage}% or more of companies): {topics_to_keep}")
    print(f"Topics to Remove: {topics_to_remove}")
    print(f"Percentage Threshold: {threshold_percentage}%")
    
    return topics_to_keep

def create_transition_matrix(topic_sequence, num_topics):
    transition_matrix = np.zeros((num_topics, num_topics))
    for i in range(len(topic_sequence) - 1):
        from_topic = topic_sequence[i]
        to_topic = topic_sequence[i + 1]
        transition_matrix[from_topic][to_topic] += 1
    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    return transition_matrix

def compute_similarity_to_average(df, num_topics):
    # Create transition matrices for each call
    transition_matrices = []
    call_ids = []
    siccds = []
    permcos = []
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        if len(topics) < 2:
            # Cannot create a transition matrix with fewer than 2 topics
            continue
        transition_matrix = create_transition_matrix(topics, num_topics)
        transition_matrices.append(transition_matrix)
        call_ids.append(call_id)
        siccd = group['siccd'].iloc[0]  # Assuming 'siccd' is consistent within a call
        permco = group['permco'].iloc[0]  # Assuming 'permco' is consistent within a call
        siccds.append(siccd)
        permcos.append(permco)

    # Create a DataFrame to hold the data
    calls_df = pd.DataFrame({
        'call_id': call_ids,
        'transition_matrix': transition_matrices,
        'siccd': siccds,
        'permco': permcos
    })

    # Compute the overall average transition matrix
    average_transition_matrix = np.mean(transition_matrices, axis=0)
    average_transition_matrix = np.nan_to_num(average_transition_matrix)

    # Compute industry-specific average transition matrices
    industry_avg_matrices = {}
    for siccd, group in calls_df.groupby('siccd'):
        if pd.isna(siccd):
            continue  # Skip missing siccds
        matrices = group['transition_matrix'].tolist()
        if matrices:
            industry_avg_matrix = np.mean(matrices, axis=0)
            industry_avg_matrix = np.nan_to_num(industry_avg_matrix)
            industry_avg_matrices[siccd] = industry_avg_matrix

    # Compute company-specific average transition matrices
    company_avg_matrices = {}
    for permco, group in calls_df.groupby('permco'):
        matrices = group['transition_matrix'].tolist()
        if matrices:
            company_avg_matrix = np.mean(matrices, axis=0)
            company_avg_matrix = np.nan_to_num(company_avg_matrix)
            company_avg_matrices[permco] = company_avg_matrix

    # Compute similarities
    similarities_overall = []
    similarities_industry = []
    similarities_company = []
    for idx, row in calls_df.iterrows():
        tm = np.nan_to_num(row['transition_matrix'])
        tm_vector = tm.flatten()
        # Similarity to overall average
        avg_vector = average_transition_matrix.flatten()
        # Check if both vectors are non-zero
        if np.linalg.norm(tm_vector) == 0 or np.linalg.norm(avg_vector) == 0:
            sim_overall = np.nan
        else:
            sim_overall = 1 - cosine(tm_vector, avg_vector)
        similarities_overall.append(sim_overall)
        # Similarity to industry average
        if pd.isna(row['siccd']):
            sim_industry = np.nan
        else:
            industry_avg_matrix = industry_avg_matrices.get(row['siccd'])
            if industry_avg_matrix is not None:
                industry_avg_vector = industry_avg_matrix.flatten()
                if np.linalg.norm(industry_avg_vector) == 0 or np.linalg.norm(tm_vector) == 0:
                    sim_industry = np.nan
                else:
                    sim_industry = 1 - cosine(tm_vector, industry_avg_vector)
            else:
                sim_industry = np.nan
        similarities_industry.append(sim_industry)
        # Similarity to company average
        company_avg_matrix = company_avg_matrices[row['permco']]
        company_avg_vector = company_avg_matrix.flatten()
        if np.linalg.norm(company_avg_vector) == 0 or np.linalg.norm(tm_vector) == 0:
            sim_company = np.nan
        else:
            sim_company = 1 - cosine(tm_vector, company_avg_vector)
        similarities_company.append(sim_company)

    # Add similarities to calls_df
    calls_df['similarity_to_overall_average'] = similarities_overall
    calls_df['similarity_to_industry_average'] = similarities_industry
    calls_df['similarity_to_company_average'] = similarities_company

    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']]
def compute_similarity_to_average(df, num_topics):
    # Create transition matrices for each call and include call_date
    transition_matrices = []
    call_ids = []
    siccds = []
    permcos = []
    call_dates = []
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        if len(topics) < 2:
            # Cannot create a transition matrix with fewer than 2 topics
            continue
        transition_matrix = create_transition_matrix(topics, num_topics)
        transition_matrices.append(transition_matrix)
        call_ids.append(call_id)
        siccd = group['siccd'].iloc[0]  # Assuming 'siccd' is consistent within a call
        permco = group['permco'].iloc[0]  # Assuming 'permco' is consistent within a call
        call_date = group['call_date'].iloc[0]
        siccds.append(siccd)
        permcos.append(permco)
        call_dates.append(call_date)

    # Create a DataFrame to hold the data
    calls_df = pd.DataFrame({
        'call_id': call_ids,
        'transition_matrix': transition_matrices,
        'siccd': siccds,
        'permco': permcos,
        'call_date': call_dates
    })

    # Sort calls_df by 'call_date'
    calls_df = calls_df.sort_values('call_date').reset_index(drop=True)

    # Initialize lists to store similarities
    similarities_overall = []
    similarities_industry = []
    similarities_company = []

    # Iterate over each call to compute similarities based on prior data
    for idx, row in calls_df.iterrows():
        current_call_date = row['call_date']
        current_call_id = row['call_id']
        tm = np.nan_to_num(row['transition_matrix'])
        tm_vector = tm.flatten()

        # Get previous calls (before current_call_date)
        previous_calls_df = calls_df[calls_df['call_date'] < current_call_date]

        # If there are no previous calls, similarities will be NaN
        if previous_calls_df.empty:
            sim_overall = np.nan
            sim_industry = np.nan
            sim_company = np.nan
        else:
            # Compute overall average transition matrix from previous calls
            previous_matrices = previous_calls_df['transition_matrix'].tolist()
            average_transition_matrix = np.mean(previous_matrices, axis=0)
            average_transition_matrix = np.nan_to_num(average_transition_matrix)
            avg_vector = average_transition_matrix.flatten()

            # Compute similarity to overall average
            if np.linalg.norm(tm_vector) == 0 or np.linalg.norm(avg_vector) == 0:
                sim_overall = np.nan
            else:
                sim_overall = 1 - cosine(tm_vector, avg_vector)

            # Compute industry average transition matrix from previous calls within the same industry
            siccd = row['siccd']
            if pd.isna(siccd):
                sim_industry = np.nan
            else:
                previous_industry_calls_df = previous_calls_df[previous_calls_df['siccd'] == siccd]
                if previous_industry_calls_df.empty:
                    sim_industry = np.nan
                else:
                    previous_industry_matrices = previous_industry_calls_df['transition_matrix'].tolist()
                    industry_avg_matrix = np.mean(previous_industry_matrices, axis=0)
                    industry_avg_matrix = np.nan_to_num(industry_avg_matrix)
                    industry_avg_vector = industry_avg_matrix.flatten()
                    if np.linalg.norm(tm_vector) == 0 or np.linalg.norm(industry_avg_vector) == 0:
                        sim_industry = np.nan
                    else:
                        sim_industry = 1 - cosine(tm_vector, industry_avg_vector)

            # Compute company average transition matrix from previous calls within the same company
            permco = row['permco']
            previous_company_calls_df = previous_calls_df[previous_calls_df['permco'] == permco]
            if previous_company_calls_df.empty:
                sim_company = np.nan
            else:
                previous_company_matrices = previous_company_calls_df['transition_matrix'].tolist()
                company_avg_matrix = np.mean(previous_company_matrices, axis=0)
                company_avg_matrix = np.nan_to_num(company_avg_matrix)
                company_avg_vector = company_avg_matrix.flatten()
                if np.linalg.norm(tm_vector) == 0 or np.linalg.norm(company_avg_vector) == 0:
                    sim_company = np.nan
                else:
                    sim_company = 1 - cosine(tm_vector, company_avg_vector)

        # Append similarities to lists
        similarities_overall.append(sim_overall)
        similarities_industry.append(sim_industry)
        similarities_company.append(sim_company)

    # Add similarities to calls_df
    calls_df['similarity_to_overall_average'] = similarities_overall
    calls_df['similarity_to_industry_average'] = similarities_industry
    calls_df['similarity_to_company_average'] = similarities_company

    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']]
    return similarity_df
