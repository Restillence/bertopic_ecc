# utils.py

import json
from bertopic import BERTopic
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
    Determine topics that appear in at least 1 call for the specified percentage of companies,
    excluding the outlier category (-1).
    """
    # Get the total number of companies
    total_companies = df['permco'].nunique()

    # Set to keep track of topics per company
    topics_per_company = df.groupby('permco')['topics'].apply(lambda x: set().union(*x)).reset_index()

    # Count the occurrences of each topic across companies
    topic_counts = {}

    for topics in topics_per_company['topics']:
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
    import numpy as np
    from scipy.spatial.distance import cosine
    import pandas as pd

    # Create transition matrices for each call and include call_date
    transition_matrices = []
    call_ids = []
    siccds = []
    permcos = []
    call_dates = []

    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = group['filtered_topics'].values[0]  # Get the list of topics
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
    overall_mean = np.zeros(num_elements, dtype=np.float64)
    count = 0

    for i in range(len(calls_df)):
        tm_vector = calls_df.iloc[i][tm_flat_df.columns].values.astype(np.float64)
        if count > 0:
            similarity = 1 - cosine(tm_vector, overall_mean)
        else:
            similarity = np.nan
        similarities_overall.append(similarity)
        # Update overall mean
        count += 1
        overall_mean += (tm_vector - overall_mean) / count

    calls_df['similarity_to_overall_average'] = similarities_overall

    ### Compute similarities to industry average ###
    similarities_industry = []
    running_means = {}
    counts = {}

    # Sort calls_df by 'siccd' and 'call_date'
    calls_df_industry = calls_df.sort_values(['siccd', 'call_date']).reset_index(drop=True)

    for i in range(len(calls_df_industry)):
        siccd = calls_df_industry['siccd'].iloc[i]
        tm_vector = calls_df_industry.iloc[i][tm_flat_df.columns].values.astype(np.float64)

        if siccd not in running_means:
            running_means[siccd] = np.zeros(num_elements, dtype=np.float64)
            counts[siccd] = 0

        if counts[siccd] > 0:
            mean_vector = running_means[siccd]
            similarity = 1 - cosine(tm_vector, mean_vector)
        else:
            similarity = np.nan
        similarities_industry.append(similarity)

        # Update running mean for the industry
        counts[siccd] += 1
        running_means[siccd] += (tm_vector - running_means[siccd]) / counts[siccd]

    calls_df_industry['similarity_to_industry_average'] = similarities_industry

    ### Compute similarities to company average ###
    similarities_company = []
    running_means_company = {}
    counts_company = {}

    # Sort calls_df by 'permco' and 'call_date'
    calls_df_company = calls_df.sort_values(['permco', 'call_date']).reset_index(drop=True)

    for i in range(len(calls_df_company)):
        permco = calls_df_company['permco'].iloc[i]
        tm_vector = calls_df_company.iloc[i][tm_flat_df.columns].values.astype(np.float64)

        if permco not in running_means_company:
            running_means_company[permco] = np.zeros(num_elements, dtype=np.float64)
            counts_company[permco] = 0

        if counts_company[permco] > 0:
            mean_vector = running_means_company[permco]
            similarity = 1 - cosine(tm_vector, mean_vector)
        else:
            similarity = np.nan
        similarities_company.append(similarity)

        # Update running mean for the company
        counts_company[permco] += 1
        running_means_company[permco] += (tm_vector - running_means_company[permco]) / counts_company[permco]

    calls_df_company['similarity_to_company_average'] = similarities_company

    # Merge the similarities back into the original calls_df
    calls_df = calls_df.merge(
        calls_df_industry[['call_id', 'similarity_to_industry_average']],
        on='call_id',
        how='left'
    )

    calls_df = calls_df.merge(
        calls_df_company[['call_id', 'similarity_to_company_average']],
        on='call_id',
        how='left'
    )

    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']]
    return similarity_df
