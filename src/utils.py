# utils.py

import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import ast
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import dok_matrix
import numpy as np

def print_configuration(config):
    """
    Print the configuration dictionary.
    """
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def count_word_length_text(texts):
    """
    Count the total number of words in the 'presentation_text' list for a single row.
    """
    if not isinstance(texts, list):
        return 0
    total_words = 0
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            total_words += len(words)
    return total_words

def count_items(items):
    """
    Count the number of items in a list for a single row.
    """
    if not isinstance(items, list):
        return 0
    return len(items)

def process_topics(path, output_path, topics_to_keep, threshold_percentage=None):
    """
    Process the topics by filtering based on the specified criteria.
    """
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Check if 'presentation_topics' column exists
    if 'presentation_topics' not in df.columns:
        raise KeyError("'presentation_topics' column not found in the input data.")
    
    if topics_to_keep == "all":
        df['filtered_presentation_topics'] = df["presentation_topics"]
        df['filtered_texts'] = df["presentation_text"]
        df.to_csv(output_path, index=False)
        return df
    
    # Convert string representations of lists into actual lists using ast.literal_eval
    df['presentation_topics'] = df['presentation_topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['presentation_text'] = df['presentation_text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Parse 'ceo_names' and 'cfo_names' using ast.literal_eval
    df['ceo_names'] = df['ceo_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df['cfo_names'] = df['cfo_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    # Debug: Print sample parsed 'ceo_names' and 'cfo_names'
    print("Sample parsed 'ceo_names' and 'cfo_names':")
    print(df[['ceo_names', 'cfo_names']].head())
    
    # **Ensure topics are integers**
    try:
        df['presentation_topics'] = df['presentation_topics'].apply(lambda topics: [int(t) for t in topics])
    except ValueError as e:
        print(f"Error converting presentation_topics to integers: {e}")
        df['presentation_topics'] = [[] for _ in range(len(df))]
    
    # Debug: Print unique topics in 'presentation_topics'
    unique_presentation_topics = set().union(*df['presentation_topics'])
    print(f"Unique topics in 'presentation_topics': {sorted(unique_presentation_topics)}")
    
    # If topics_to_keep is "auto", determine topics to keep
    if topics_to_keep == "auto":
        topics_to_keep = determine_topics_to_keep(df, threshold_percentage)
    else:
        # **Convert topics_to_keep from strings to integers if necessary**
        if isinstance(topics_to_keep, list):
            try:
                topics_to_keep = set(int(t) for t in topics_to_keep)
                print(f"Topics to keep (converted to integers): {sorted(topics_to_keep)}")
            except ValueError as e:
                print(f"Error converting topics_to_keep to integers: {e}")
                topics_to_keep = set()
        else:
            print(f"Invalid type for topics_to_keep: {type(topics_to_keep)}. Expected list or 'auto'.")
            topics_to_keep = set()
    
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
    df[['filtered_presentation_topics', 'filtered_texts']] = df.apply(
        lambda row: keep_presentation_topics_and_texts(row, topics_to_keep),
        axis=1, result_type='expand')
    
    # Debug: Check how many rows have non-empty 'filtered_presentation_topics'
    non_empty_filtered = df['filtered_presentation_topics'].apply(len).gt(0).sum()
    print(f"Number of rows with non-empty 'filtered_presentation_topics': {non_empty_filtered} out of {len(df)}")
    
    # Debug: Print unique topics after filtering
    if non_empty_filtered > 0:
        unique_filtered_topics = set().union(*df['filtered_presentation_topics'])
        print(f"Unique topics after filtering: {sorted(unique_filtered_topics)}")
    else:
        print("All 'filtered_presentation_topics' are empty.")
    
    # Debug: Inspect first few rows of filtered topics
    print("Sample 'filtered_presentation_topics' after filtering:")
    print(df['filtered_presentation_topics'].head())
    
    # Consistency check to validate if presentation_topics and texts are of the same length
    def check_consistency(row):
        return len(row['presentation_topics']) == len(row['presentation_text'])
    
    df['consistent'] = df.apply(check_consistency, axis=1)
    
    # Save the processed topics to the output path
    df.to_csv(output_path, index=False)
    
    # Return relevant columns, including the new ones
    return df[['presentation_topics', 'presentation_text', 'filtered_presentation_topics', 'filtered_texts', 'consistent', 'call_id', 'permco', 'date',
               'ceo_participates', 'ceo_names', 'cfo_names', 'participant_question_topics', 'management_answer_topics']]

def determine_topics_to_keep(df, threshold_percentage):
    """
    Determine topics that appear in at least a certain percentage of companies,
    excluding the outlier category (-1).
    """
    # Get the total number of companies
    total_companies = df['permco'].nunique()
    print(f"Total number of unique companies (permco): {total_companies}")
    
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
    print(f"Company threshold (number of companies): {company_threshold}")
    
    # Find topics that appear in at least the threshold percentage of companies
    topics_to_keep = {topic for topic, count in topic_counts.items() if count >= company_threshold}
    
    # Find topics to remove
    topics_to_remove = set(topic_counts.keys()) - topics_to_keep
    
    # Print statements to show kept and removed topics
    print(f"Topics to Keep (Appearing in {threshold_percentage}% or more of companies): {sorted(topics_to_keep)}")
    print(f"Topics to Remove: {sorted(topics_to_remove)}")
    print(f"Percentage Threshold: {threshold_percentage}%")
    
    return topics_to_keep

def create_transition_matrix(topics_list, num_topics):
    """
    Creates a transition matrix from a list of topic sequences.

    Parameters:
    - topics_list (list of lists): Each sublist represents a sequence of topics in a presentation.
    - num_topics (int): Total number of unique topics.

    Returns:
    - transition_matrix (numpy.ndarray): A matrix of shape (num_topics, num_topics) where each cell [i][j]
      represents the count of transitions from topic i to topic j.
    """
    transition_matrix = np.zeros((num_topics, num_topics), dtype=int)

    for topics in tqdm(topics_list, desc="Creating Transition Matrix"):
        # Ensure that topics is a list with at least two elements
        if isinstance(topics, list) and len(topics) >= 2:
            for i in range(len(topics) - 1):
                from_topic = topics[i]
                to_topic = topics[i + 1]
                if isinstance(from_topic, int) and isinstance(to_topic, int):
                    if 0 <= from_topic < num_topics and 0 <= to_topic < num_topics:
                        transition_matrix[from_topic][to_topic] += 1

    # Debug: Print transition matrix summary
    print(f"Transition Matrix Summary:\n{transition_matrix}")
    print(f"Transition Matrix Shape: {transition_matrix.shape}")

    return transition_matrix

def compute_similarity_to_average(df, num_topics):
    """
    Compute similarity measures to overall, industry, and company averages.
    """
    # Create lists to hold the data
    call_ids = []
    siccds = []
    permcos = []
    call_dates = []
    transition_vectors = []

    # Ensure 'call_date' is in datetime format
    df['call_date'] = pd.to_datetime(df['call_date'])

    # Create transition vectors for each call with progress bar
    print("Creating transition vectors...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Calls"):
        call_id = row['call_id']
        topics = row['filtered_presentation_topics']
        if len(topics) < 2:
            continue  # Cannot create a transition matrix with fewer than 2 topics
        tm = create_transition_matrix(topics, num_topics)
        # Flatten the matrix into a vector
        tm_vector = tm.toarray().flatten()
        transition_vectors.append(tm_vector)
        call_ids.append(call_id)
        siccds.append(row['siccd'])
        permcos.append(row['permco'])
        call_dates.append(row['call_date'])

    # Create a DataFrame
    calls_df = pd.DataFrame({
        'call_id': call_ids,
        'transition_vector': transition_vectors,
        'siccd': siccds,
        'permco': permcos,
        'call_date': call_dates
    })

    # Sort calls_df by 'call_date'
    calls_df = calls_df.sort_values('call_date').reset_index(drop=True)

    # Compute similarities with progress bar
    similarities_overall = []
    similarities_industry = []
    similarities_company = []

    print("Computing similarities...")
    for idx, row in tqdm(calls_df.iterrows(), total=calls_df.shape[0], desc="Computing Similarities"):
        current_date = row['call_date']
        tm_vector = row['transition_vector'].reshape(1, -1)  # Reshape for sklearn

        # Define the time window: previous 1 year (365 days) + 20 days buffer
        time_window_start = current_date - pd.DateOffset(days=385)

        # Filter for calls within the time window and before the current call date
        window_mask = (calls_df['call_date'] >= time_window_start) & (calls_df['call_date'] < current_date)

        # Overall similarity
        window_df = calls_df[window_mask]
        if len(window_df) < 4:
            similarities_overall.append(np.nan)
        else:
            # Compute the average transition vector
            mean_vector = np.mean(np.vstack(window_df['transition_vector'].values), axis=0).reshape(1, -1)
            # Compute similarity
            similarity = cosine_similarity(tm_vector, mean_vector)[0][0]
            similarities_overall.append(similarity)

        # Industry similarity
        industry_df = window_df[window_df['siccd'] == row['siccd']]
        if len(industry_df) < 4:
            similarities_industry.append(np.nan)
        else:
            mean_vector = np.mean(np.vstack(industry_df['transition_vector'].values), axis=0).reshape(1, -1)
            similarity = cosine_similarity(tm_vector, mean_vector)[0][0]
            similarities_industry.append(similarity)

        # Company similarity
        company_df = calls_df[(calls_df['permco'] == row['permco']) & (calls_df['call_date'] < current_date)].tail(4)
        if len(company_df) < 4:
            similarities_company.append(np.nan)
        else:
            mean_vector = np.mean(np.vstack(company_df['transition_vector'].values), axis=0).reshape(1, -1)
            similarity = cosine_similarity(tm_vector, mean_vector)[0][0]
            similarities_company.append(similarity)

    # Add similarities to calls_df
    calls_df['similarity_to_overall_average'] = similarities_overall
    calls_df['similarity_to_industry_average'] = similarities_industry
    calls_df['similarity_to_company_average'] = similarities_company

    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']]
    return similarity_df

# Define Zeroshot Clusters
zeroshot_clusters = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [163, 166],
    15: [123, 63, 46],
    16: [181, 207, 97, 105, 83, 186, 112],
    17: [53, 115, 84, 111, 126, 15, 214, 191, 157, 77, 122, 146],
    18: [170, 205, 201, 130, 16, 118, 125, 33, 22, 152, 90, 145, 18,188],
    19: [39, 74, 200],
    20: [137, 187],
    21: [45, 60, 19],
    22: [155, 57],
    23: [113, 52, 213],
    24: [220, 199, 219, 66, 25, 81, 140, 124, 64, 120, 159, 136, 103],
    25: [91, 174, 29, 38],
    26: [98, 23, 208],
    27: [114, 109, 14, 203],
    28: [151, 42],
    29: [147, 177, 59, 169],
    30: [164, 89, 218, 180],
    31: [62, 202, 206, 55, 129, 212, 121, 31, 95, 116, 37, 86, 54, 108, 71],
    32: [87, 156],
    33: [72, 75, 161, 34, 47, 20],
    34: [80, 185],
    35: [92, 217, 134, 128, 204, 168],
    36: [142, 78, 51, 76, 197, 93, 35, 154],
    37: [194, 40, 41, 179],
    38: [88, 110, 106, 79, 196],
    39: [135, 148, 67],
    40: [183, 99, 27],
    41: [70, 117],
    42: [131, 69],
    43: [102, 192, 210],
    44: [32, 104, 73, 21],
    45: [58, 85],
    46: [96, 132, 119, 65],
    47: [24, 94, 216, 17],
    48: [149, 36],
    49: [176, 184, 28],
    50: [101, 167, 158],
    51: [43, 82, 56, 61, 48],
    52: [182, 165, 68, 162, 44],
    53: [107, 127],
    54: [171, 144, 178, 133, 100, 215],
    55: [173, 175, 153, 172, 49, 141, 193, 30, 26],
    56: [195, 143, 50, 189, 190, 150, 198, 139, 138, 211, 209, 160]
}

# Define Regular (BERTopic) Clusters
regular_clusters = {
    0: [418, 69, 22, 379, 237],
    1: [480, 259, 214, 144, 58, 178],
    2: [490, 412, 364, 90, 495, 64],
    3: [353, 385, 358, 410],
    4: [291, 75, 74, 450, 347],
    5: [189, 9, 469, 102, 136, 265, 38],
    6: [389, 307, 140, 307, 261],
    7: [360, 310, 502, 104, 484, 161, 190, 288],
    8: [496, 157, 177, 432, 452, 467, 463, 498, 240],
    9: [376, 322],
    10: [363, 234, 43, 437, 52],
    11: [321, 460, 333, 142, 124, 311, 8, 319],
    12: [83, 422, 66, 101, 269, 213, 372],
    13: [295, 54, 448],
    14: [257, 206, 454, 476, 17, 173, 154, 122, 332, 107],
    15: [421, 20],
    16: [451, 184, 280, 229, 191],
    17: [274, 270, 227, 371, 457, 277, 462, 263, 382, 493, 323, 330],
    18: [483, 449, 474, 133, 215, 207, 504, 468, 134, 174],
    19: [424, 255, 428, 482, 340, 99, 386, 489, 497, 325, 84, 199, 106, 68, 252, 267, 505, 387],
    20: [192, 159, 33, 63, 337, 328],
    21: [479, 148, 67, 187, 130, 108, 361, 193, 390],
    22: [248, 494, 6, 470, 458, 503, 79, 268],
    23: [380, 298, 209, 220, 315, 73, 0, 431, 312, 426, 417, 217, 306, 86, 155, 341, 135, 287, 366, 413, 29, 296],
    24: [116, 357, 301, 471, 420],
    25: [201, 153, 342, 93, 396, 406, 222, 442, 208, 128, 197, 313],
    26: [243, 179, 300, 314, 430],
    27: [282, 244, 204, 272, 461, 381, 176, 126],
    28: [256, 150, 127, 160, 441, 180, 440],
    29: [293, 89, 292, 464, 109],
    30: [96, 94, 289, 445, 501],
    31: [486, 351, 13, 235],
    32: [438, 499, 407],
    33: [326, 113, 4, 338],
    34: [105, 339, 23, 61, 472],
    35: [477, 149, 228],
    36: [231, 305, 31, 488],
    37: [147, 394],
    38: [175, 166],
    39: [200, 392, 434, 279],
    40: [186, 169, 15, 185, 423, 433, 398, 393, 375],
    41: [349, 278],
    42: [356, 254, 446, 36],
    43: [138, 405, 427, 336, 34, 210],
    44: [32, 246],
    45: [334, 242, 230],
    46: [478, 91, 439, 362, 500, 250],
    47: [377, 216, 455, 399],
    48: [408, 473, 402, 266, 57, 264],
    49: [238, 370],
    50: [429, 369, 60, 98, 183],
    51: [28, 132],
    52: [226, 447, 171, 327],
    53: [299, 165, 181, 404, 374],
    54: [71, 354, 56, 403, 137],
    55: [359, 3, 284, 436, 481, 453],
    56: [62, 219],
    57: [241, 151, 401, 409, 72],
    58: [290, 95, 115],
    59: [167, 415],
    60: [114, 152],
    61: [335, 172, 487],
    62: [343, 475, 456],
    63: [303, 416, 156, 212],
    64: [162, 346, 88, 2],
    65: [223, 145, 400],
    66: [397, 352],
    67: [117, 39, 129, 203],
    68: [221, 395, 35, 7, 131, 491, 459, 25, 194, 211, 304],
    69: [249, 286, 232, 19, 271, 348, 70, 18, 123, 196],
    70: [435, 87, 320, 224, 281, 11, 164],
    71: [308, 78, 41, 365, 466, 253],
    72: [425, 317, 239, 273, 82, 37, 111, 384, 40, 236, 139, 85, 373],
    73: [391, 47, 318],
    74: [188, 14, 146, 419],
    75: [329, 260, 65, 30, 195],
    76: [294, 80, 100, 44, 76],
    77: [316, 48, 331, 506, 51, 21],
    78: [125, 275, 12, 383, 53, 1],
    79: [182, 465, 81, 414],
    80: [163, 120, 285, 168, 10, 55, 444, 46, 367],
    81: [121, 262, 5, 59, 77],
    82: [378, 16, 345, 143, 92, 45, 245, 368, 97],
    83: [233, 344, 119, 112],
    84: [198, 350, 141],
    85: [388, 218, 170, 355, 324],
    86: [276, 202, 26, 24, 225, 49, 205, 110, 411, 283, 258, 247],
    87: [443, 309, 492, 50, 27, 297],
    88: [118, 103, 485],
    89: [158, 302, 42, 251]
}

def map_topics_to_clusters(df, model='zeroshot', apply_mapping=True):
    """
    Maps topic numbers in specified columns to their corresponding cluster numbers based on the model.
    Applies mapping only if apply_mapping is True.
    """
    if not apply_mapping:
        print("Skipping topic mapping as 'apply_mapping' is set to False.")
        return df
    
    # Select the appropriate cluster dictionary based on the model
    if model.lower() == 'zeroshot':
        cluster_dict = zeroshot_clusters
    elif model.lower() == 'regular':
        cluster_dict = regular_clusters
    else:
        raise ValueError("Invalid model type. Choose 'zeroshot' or 'regular'.")
    
    # Invert the cluster dictionary to map topic to cluster
    topic_to_cluster = {}
    for cluster_num, topics in cluster_dict.items():
        for topic in topics:
            topic_to_cluster[topic] = cluster_num
    
    # Define the columns to process
    topic_columns = ['filtered_presentation_topics', 'participant_question_topics', 'management_answer_topics']
    
    def replace_topics_with_clusters(topics):
        """
        Replaces a list of topic numbers with their cluster numbers.
        If a topic isn't found in any cluster, it remains unchanged.
        """
        return [topic_to_cluster.get(topic, topic) for topic in topics]
    
    # Apply the mapping to each specified column
    for col in topic_columns:
        if col in df.columns:
            # Ensure that the column contains lists
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            # Apply the mapping
            df[col] = df[col].apply(replace_topics_with_clusters)
            print(f"Mapped topics in column '{col}' to clusters.")
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    
    return df

def remove_neg_one_from_columns(df, columns):
    """
    Removes all occurrences of -1 from the specified list-type columns in the DataFrame.
    """
    for col in columns:
        if col in df.columns:
            # Attempt to convert string representations of lists to actual lists
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Ensure that the column contains lists
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            
            # Remove -1 from each list
            df[col] = df[col].apply(lambda x: [item for item in x if item != -1])
            print(f"Removed all -1s from column '{col}'.")
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    return df

def create_average_transition_matrix_figures(transition_matrix, title, output_path):
    """
    Creates and saves a heatmap figure for the average transition matrix.

    Parameters:
    - transition_matrix (numpy.ndarray): The transition matrix to visualize.
    - title (str): The title of the heatmap.
    - output_path (str): The file path to save the heatmap image.

    Returns:
    - None
    """
    # Convert the transition_matrix to a dense format if it's sparse
    if hasattr(transition_matrix, 'toarray'):
        transition_matrix = transition_matrix.toarray()
    else:
        transition_matrix = np.array(transition_matrix)

    # Calculate the average transitions (normalize rows to sum to 1)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # To avoid division by zero, set row sums that are zero to one
    row_sums[row_sums == 0] = 1
    average_transition_matrix = transition_matrix / row_sums

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 16))
    sns.heatmap(average_transition_matrix, cmap='viridis', linewidths=.5, annot=False)
    plt.title(title, fontsize=20)
    plt.xlabel('To Topic', fontsize=16)
    plt.ylabel('From Topic', fontsize=16)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved transition matrix heatmap to {output_path}")

