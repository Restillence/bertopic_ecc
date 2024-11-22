# utils.py

import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
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

def count_word_length_text(texts):
    """
    Count the total number of words in the 'presentation_text' list for a single row.

    Parameters:
    - texts (list): A list of text strings.

    Returns:
    - int: Total word count.
    """
    if not isinstance(texts, list):
        return 0
    total_words = 0
    for text in texts:
        if isinstance(text, str):
            # Split the text into words using whitespace and count
            words = text.split()
            total_words += len(words)
    return total_words

def count_items(items):
    """
    Count the number of items in a list for a single row.

    Parameters:
    - items (list): A list of items (questions or answers).

    Returns:
    - int: Number of items.
    """
    if not isinstance(items, list):
        return 0
    return len(items)

def process_topics(path, output_path, topics_to_keep, threshold_percentage=None):
    """
    Process the topics by filtering based on the specified criteria.

    Parameters:
    - path (str): Path to the input CSV file containing presentation_topics and texts.
    - output_path (str): Path to save the processed CSV file.
    - topics_to_keep (set or str): Set of topics to retain or "all" to keep all topics.
    - threshold_percentage (float, optional): Percentage threshold for auto topic selection.

    Returns:
    - pd.DataFrame: Processed DataFrame with filtered topics and texts.
    """
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Check if 'presentation_topics' column exists
    if 'presentation_topics' not in df.columns:
        raise KeyError("'presentation_topics' column not found in the input data.")
    
    if topics_to_keep == "all":
        df['filtered_presentation_topics'] = df["presentation_topics"]
        df['filtered_texts'] = df["text"]
        df.to_csv(output_path, index=False)
        return df
    
    # Convert string representations of lists into actual lists using ast.literal_eval
    df['presentation_topics'] = df['presentation_topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['presentation_text'] = df['presentation_text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Parse 'ceo_names' and 'cfo_names' using ast.literal_eval
    df['ceo_names'] = df['ceo_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df['cfo_names'] = df['cfo_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    # Validate parsing
    print("Sample parsed 'ceo_names' and 'cfo_names':")
    print(df[['ceo_names', 'cfo_names']].head())
    
    # Continue with existing processing...
    if topics_to_keep == "auto":
        topics_to_keep = determine_topics_to_keep(df, threshold_percentage)
    
    # Function to keep only the specified topics and corresponding texts
    def keep_presentation_topics_and_texts(row, topics_to_keep):
        topics = row['presentation_topics']
        texts = row['presentation_text']
        filtered_data = [(topic, text) for topic, text in zip(topics, texts) if topic in topics_to_keep]
        if filtered_data:
            filtered_presentation_topics, filtered_texts = zip(*filtered_data)
        else:
            filtered_presentation_topics, filtered_texts = [], []
        return list(filtered_presentation_topics), list(filtered_texts)
    
    # Apply the function to each row
    df[['filtered_presentation_topics', 'filtered_texts']] = df.apply(lambda row: keep_presentation_topics_and_texts(row, topics_to_keep), axis=1, result_type='expand')
    
    # Consistency check to validate if presentation_topics and texts are of the same length
    def check_consistency(row):
        return len(row['presentation_topics']) == len(row['presentation_text'])
    
    df['consistent'] = df.apply(check_consistency, axis=1)
    
    # Save the processed topics to the output path
    df.to_csv(output_path, index=False)
    
    # Return relevant columns, including the new ones
    return df[['presentation_topics', 'presentation_text', 'filtered_presentation_topics', 'filtered_texts', 'consistent', 'call_id', 'permco', 'date',
               'ceo_participates', 'ceo_names', 'cfo_names','participant_question_topics', 'management_answer_topics']]

def determine_topics_to_keep(df, threshold_percentage):
    """
    Determine topics that appear in at least a certain percentage of companies,
    excluding the outlier category (-1).

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'permco' and 'presentation_topics' columns.
    - threshold_percentage (float): Percentage threshold for keeping topics.

    Returns:
    - set: Set of topics to keep.
    """
    # Get the total number of companies
    total_companies = df['permco'].nunique()

    # Set to keep track of topics per company
    topics_per_company = df.groupby('permco')['presentation_topics'].apply(lambda x: set().union(*x)).reset_index()

    # Count the occurrences of each topic across companies
    topic_counts = {}

    for topics in topics_per_company['presentation_topics']:
        for topic in topics:
            if topic != -1:  # Exclude outlier category
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
    """
    Create a transition matrix from a sequence of topics.

    Parameters:
    - topic_sequence (list): List of topic IDs representing the sequence.
    - num_topics (int): Total number of topics.

    Returns:
    - np.ndarray: Transition matrix of shape (num_topics, num_topics).
    """
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
    """
    Compute similarity measures to overall, industry, and company averages.

    Parameters:
    - df (pd.DataFrame): DataFrame containing transition matrices and related metadata.
    - num_topics (int): Total number of topics.

    Returns:
    - pd.DataFrame: DataFrame containing similarity measures.
    """
    # Create transition matrices for each call and include call_date
    transition_matrices = []
    call_ids = []
    siccds = []
    permcos = []
    call_dates = []

    # Ensure 'call_date' is in datetime format
    df['call_date'] = pd.to_datetime(df['call_date'])

    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = group['filtered_presentation_topics'].values[0]  # Get the list of topics
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

    # Flatten the transition matrices and create a DataFrame
    num_elements = num_topics * num_topics
    tm_flat_list = [tm.flatten() for tm in calls_df['transition_matrix']]
    tm_flat_df = pd.DataFrame(tm_flat_list, columns=[f'tm_{i}' for i in range(num_elements)], dtype=np.float64)

    # Concatenate calls_df and tm_flat_df
    calls_df = pd.concat([calls_df.reset_index(drop=True), tm_flat_df.reset_index(drop=True)], axis=1)

    ### Compute similarities to overall average ###
    similarities_overall = []

    # Convert 'call_date' to datetime if not already
    calls_df['call_date'] = pd.to_datetime(calls_df['call_date'])

    for idx, row in calls_df.iterrows():
        current_date = row['call_date']
        tm_vector = row[tm_flat_df.columns].values.astype(np.float64)

        # Define the time window: previous 1 year (365 days) + 20 additional days for buffer
        time_window_start = current_date - pd.DateOffset(days=365+20)

        # Filter for calls within the time window and before the current call date
        window_df = calls_df[(calls_df['call_date'] >= time_window_start) &
                             (calls_df['call_date'] < current_date)]

        if len(window_df) < 4:
            similarity = np.nan
        else:
            # Compute the average transition matrix
            mean_vector = window_df[tm_flat_df.columns].mean().values.astype(np.float64)
            # Compute similarity
            similarity = 1 - cosine(tm_vector, mean_vector)
        similarities_overall.append(similarity)

    calls_df['similarity_to_overall_average'] = similarities_overall

    ### Compute similarities to industry average ###
    similarities_industry = []

    for idx, row in calls_df.iterrows():
        current_date = row['call_date']
        tm_vector = row[tm_flat_df.columns].values.astype(np.float64)
        siccd = row['siccd']

        # Define the time window: previous 1 year + 20 additional days for buffer
        time_window_start = current_date - pd.DateOffset(days=365+20)

        # Filter for calls within the time window, same industry, and before the current call date
        window_df = calls_df[(calls_df['call_date'] >= time_window_start) &
                             (calls_df['call_date'] < current_date) &
                             (calls_df['siccd'] == siccd)]

        if len(window_df) < 4:
            similarity = np.nan
        else:
            # Compute the average transition matrix
            mean_vector = window_df[tm_flat_df.columns].mean().values.astype(np.float64)
            # Compute similarity
            similarity = 1 - cosine(tm_vector, mean_vector)
        similarities_industry.append(similarity)

    calls_df['similarity_to_industry_average'] = similarities_industry

    ### Compute similarities to company average ###
    similarities_company = []

    for idx, row in calls_df.iterrows():
        current_date = row['call_date']
        tm_vector = row[tm_flat_df.columns].values.astype(np.float64)
        permco = row['permco']

        # Filter for previous 4 calls of the same company before the current call date
        company_calls = calls_df[(calls_df['permco'] == permco) &
                                 (calls_df['call_date'] < current_date)].sort_values('call_date', ascending=False)

        if len(company_calls) < 4:
            similarity = np.nan
        else:
            # Take the last 4 calls
            window_df = company_calls.head(4)
            # Compute the average transition matrix
            mean_vector = window_df[tm_flat_df.columns].mean().values.astype(np.float64)
            # Compute similarity
            similarity = 1 - cosine(tm_vector, mean_vector)
        similarities_company.append(similarity)

    calls_df['similarity_to_company_average'] = similarities_company

    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']]
    return similarity_df
