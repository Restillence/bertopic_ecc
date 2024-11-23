# fuzzy_graph.py

import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from utils import (
    create_transition_matrix,
    compute_similarity_to_average,
    remove_neg_one_from_columns
)
import ast
import random

# If git structure is not working properly:
fallback_config_path = "C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json"

def get_top_n_clusters(df, n=10):
    """
    Determine the top N most frequent clusters in the 'filtered_presentation_topics' column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'filtered_presentation_topics'.
    - n (int): Number of top clusters to return.
    
    Returns:
    - list: List of top N cluster numbers.
    """
    all_topics = [topic for sublist in df['filtered_presentation_topics'] for topic in sublist]
    topic_counts = pd.Series(all_topics).value_counts()
    top_n = topic_counts.head(n).index.tolist()
    print(f"Top {n} clusters based on frequency: {top_n}")
    return top_n

def filter_top_clusters(df, top_clusters):
    """
    Filter the 'filtered_presentation_topics' to include only the top clusters.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'filtered_presentation_topics'.
    - top_clusters (list): List of top cluster numbers to retain.
    
    Returns:
    - pd.DataFrame: DataFrame with filtered 'filtered_presentation_topics'.
    """
    df['filtered_presentation_topics'] = df['filtered_presentation_topics'].apply(
        lambda x: [topic for topic in x if topic in top_clusters]
    )
    return df

def get_fuzzy_topic_labels(top_clusters, config_labels):
    """
    Create a dictionary for fuzzy topic labels limited to the top clusters.
    
    Parameters:
    - top_clusters (list): List of top cluster numbers.
    - config_labels (dict): Dictionary of all fuzzy topic labels from config.
    
    Returns:
    - dict: Dictionary of labels for the top clusters.
    """
    labels = {int(k): v for k, v in config_labels.items() if int(k) in top_clusters}
    # Assign default labels to clusters without predefined labels
    for cluster in top_clusters:
        if cluster not in labels:
            labels[cluster] = f"Cluster {cluster}"
    print(f"Fuzzy Topic Labels for Top Clusters: {labels}")
    return labels

def create_average_fuzzy_graphs(df, num_topics, top_clusters, fuzzy_topic_labels):
    """
    Plot the average fuzzy graphs: overall average, one random industry, and one random company.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'filtered_presentation_topics', 'siccd', and 'permco'.
    - num_topics (int): Total number of topics.
    - top_clusters (list): List of top cluster numbers to include.
    - fuzzy_topic_labels (dict): Dictionary mapping cluster numbers to labels.
    
    Returns:
    - None
    """
    # Create cluster to index mapping
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(top_clusters)}
    index_to_cluster = {idx: cluster for cluster, idx in cluster_to_index.items()}
    
    # Overall Average
    print("Creating Overall Average Fuzzy Graph...")
    all_topics = [topic for sublist in df['filtered_presentation_topics'] for topic in sublist]
    mapped_all_topics = [cluster_to_index[topic] for topic in all_topics]
    overall_transition_matrix = create_transition_matrix(mapped_all_topics, num_topics)
    # Create graph
    G_overall = nx.DiGraph()
    for cluster in top_clusters:
        G_overall.add_node(cluster)
    for i, from_cluster in enumerate(top_clusters):
        for j, to_cluster in enumerate(top_clusters):
            weight = overall_transition_matrix[i][j]
            if weight > 0:
                G_overall.add_edge(from_cluster, to_cluster, weight=weight)
    plot_fuzzy_graph(G_overall, 'Average Fuzzy Graph for All Companies', fuzzy_topic_labels=fuzzy_topic_labels)
    
    # Random Industry
    print("Creating Average Fuzzy Graph for a Random Industry...")
    industries = df['siccd'].dropna().unique().tolist()
    if industries:
        random_industry = random.choice(industries)
        industry_df = df[df['siccd'] == random_industry]
        industry_topics = [topic for sublist in industry_df['filtered_presentation_topics'] for topic in sublist]
        mapped_industry_topics = [cluster_to_index[topic] for topic in industry_topics]
        industry_transition_matrix = create_transition_matrix(mapped_industry_topics, num_topics)
        # Create graph
        G_industry = nx.DiGraph()
        for cluster in top_clusters:
            G_industry.add_node(cluster)
        for i, from_cluster in enumerate(top_clusters):
            for j, to_cluster in enumerate(top_clusters):
                weight = industry_transition_matrix[i][j]
                if weight > 0:
                    G_industry.add_edge(from_cluster, to_cluster, weight=weight)
        plot_fuzzy_graph(G_industry, f'Average Fuzzy Graph for Industry {random_industry}', fuzzy_topic_labels=fuzzy_topic_labels)
    else:
        print("No industries found to plot.")
    
    # Random Company
    print("Creating Average Fuzzy Graph for a Random Company...")
    companies = df['permco'].dropna().unique().tolist()
    if companies:
        random_company = random.choice(companies)
        company_df = df[df['permco'] == random_company]
        company_topics = [topic for sublist in company_df['filtered_presentation_topics'] for topic in sublist]
        mapped_company_topics = [cluster_to_index[topic] for topic in company_topics]
        company_transition_matrix = create_transition_matrix(mapped_company_topics, num_topics)
        # Create graph
        G_company = nx.DiGraph()
        for cluster in top_clusters:
            G_company.add_node(cluster)
        for i, from_cluster in enumerate(top_clusters):
            for j, to_cluster in enumerate(top_clusters):
                weight = company_transition_matrix[i][j]
                if weight > 0:
                    G_company.add_edge(from_cluster, to_cluster, weight=weight)
        plot_fuzzy_graph(G_company, f'Average Fuzzy Graph for Company {random_company}', fuzzy_topic_labels=fuzzy_topic_labels)
    else:
        print("No companies found to plot.")

def plot_fuzzy_graph(graph, title, fuzzy_topic_labels=None):
    """
    Plot a fuzzy graph using NetworkX and Matplotlib.
    
    Parameters:
    - graph (networkx.DiGraph): The fuzzy graph to plot.
    - title (str): Title of the plot.
    - fuzzy_topic_labels (dict, optional): Dictionary mapping node numbers to labels.
    
    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    node_labels = {node: fuzzy_topic_labels.get(node, f'Topic {node}') for node in graph.nodes()}
    
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    # Normalize weights for visualization
    max_weight = max(weights) if weights else 1
    normalized_weights = [w / max_weight for w in weights]
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, width=[w * 5 for w in normalized_weights])
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load configuration variables from config.json
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            print("Config File Loaded.")
    except Exception as e:
        with open(fallback_config_path, 'r') as config_file:
            config = json.load(config_file)
            print("Fallback Config File Loaded.")
    
    # Define the path to the final dataset
    input_path = r'D:\daten_masterarbeit\final_dataset.csv' # Ensure it points to 'final_dataset.csv'
    
    # Define fuzzy topic labels (only top 10 clusters will be used)
    config_labels = config.get('fuzzy_topic_labels', {})
    
    # Read the final dataset
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure 'filtered_presentation_topics' is evaluated as lists
    if 'filtered_presentation_topics' in df.columns:
        df['filtered_presentation_topics'] = df['filtered_presentation_topics'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    else:
        raise ValueError("'filtered_presentation_topics' column not found in the DataFrame.")
    
    # Remove '-1' from 'participant_question_topics' and 'management_answer_topics'
    if 'participant_question_topics' in df.columns and 'management_answer_topics' in df.columns:
        df = remove_neg_one_from_columns(df, ['participant_question_topics', 'management_answer_topics'])
    else:
        print("Warning: 'participant_question_topics' or 'management_answer_topics' column not found in the DataFrame.")
    
    # Determine the top 10 most frequent clusters
    top_clusters = get_top_n_clusters(df, n=10)
    
    # Filter 'filtered_presentation_topics' to include only top clusters
    df = filter_top_clusters(df, top_clusters)
    
    # Update fuzzy topic labels to include only top clusters
    fuzzy_topic_labels = get_fuzzy_topic_labels(top_clusters, config_labels)
    
    # Compute the number of topics (top clusters)
    num_topics = len(top_clusters)
    
    # Compute similarity measures (overall, industry, company)
    print("Computing similarity measures...")
    similarity_df = compute_similarity_to_average(df, num_topics)
    
    # Merge similarity measures back into the DataFrame
    df = df.merge(similarity_df, on='call_id', how='left')
    
    # Create fuzzy graphs for each call
    print("Creating Fuzzy Graphs for Each Call...")
    fuzzy_graphs = {}
    grouped_calls = df.groupby('call_id')
    for call_id, group in grouped_calls:
        topics = group['filtered_presentation_topics'].iloc[0]
        if not topics:
            continue  # Skip if no topics
        # Create mapping from cluster to index
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(top_clusters)}
        # Map topics to indices
        try:
            mapped_topics = [cluster_to_index[topic] for topic in topics]
        except KeyError as e:
            print(f"KeyError: Cluster {e} not found in top_clusters. Skipping call_id {call_id}.")
            continue
        transition_matrix = create_transition_matrix(mapped_topics, num_topics)
        # Create graph with actual cluster numbers
        G = nx.DiGraph()
        for cluster in top_clusters:
            G.add_node(cluster)
        for i, from_cluster in enumerate(top_clusters):
            for j, to_cluster in enumerate(top_clusters):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(from_cluster, to_cluster, weight=weight)
        fuzzy_graphs[call_id] = G
    
    # Create fuzzy graphs for each industry
    print("Creating Fuzzy Graphs for Each Industry...")
    industry_fuzzy_graphs = {}
    grouped_industries = df.groupby('siccd')
    for siccd, group in grouped_industries:
        topics = [topic for sublist in group['filtered_presentation_topics'] for topic in sublist]
        if not topics:
            continue  # Skip if no topics
        # Create mapping from cluster to index
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(top_clusters)}
        # Map topics to indices
        try:
            mapped_topics = [cluster_to_index[topic] for topic in topics]
        except KeyError as e:
            print(f"KeyError: Cluster {e} not found in top_clusters. Skipping siccd {siccd}.")
            continue
        transition_matrix = create_transition_matrix(mapped_topics, num_topics)
        # Create graph
        G = nx.DiGraph()
        for cluster in top_clusters:
            G.add_node(cluster)
        for i, from_cluster in enumerate(top_clusters):
            for j, to_cluster in enumerate(top_clusters):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(from_cluster, to_cluster, weight=weight)
        industry_fuzzy_graphs[siccd] = G
    
    # Create fuzzy graphs for each company
    print("Creating Fuzzy Graphs for Each Company...")
    company_fuzzy_graphs = {}
    grouped_companies = df.groupby('permco')
    for permco, group in grouped_companies:
        topics = [topic for sublist in group['filtered_presentation_topics'] for topic in sublist]
        if not topics:
            continue  # Skip if no topics
        # Create mapping from cluster to index
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(top_clusters)}
        # Map topics to indices
        try:
            mapped_topics = [cluster_to_index[topic] for topic in topics]
        except KeyError as e:
            print(f"KeyError: Cluster {e} not found in top_clusters. Skipping permco {permco}.")
            continue
        transition_matrix = create_transition_matrix(mapped_topics, num_topics)
        # Create graph
        G = nx.DiGraph()
        for cluster in top_clusters:
            G.add_node(cluster)
        for i, from_cluster in enumerate(top_clusters):
            for j, to_cluster in enumerate(top_clusters):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(from_cluster, to_cluster, weight=weight)
        company_fuzzy_graphs[permco] = G
    
    # Plot fuzzy graphs for a specific call (example)
    if fuzzy_graphs:
        call_id_to_plot = next(iter(fuzzy_graphs))  # Get the first call_id
        graph_to_plot = fuzzy_graphs[call_id_to_plot]
        plot_fuzzy_graph(graph_to_plot, f'Fuzzy Graph for Call {call_id_to_plot}', fuzzy_topic_labels=fuzzy_topic_labels)
    else:
        print("No fuzzy graphs available to plot for any call.")
    
    # Plot fuzzy graphs for industries
    for siccd, graph in industry_fuzzy_graphs.items():
        plot_fuzzy_graph(graph, f'Fuzzy Graph for Industry {siccd}', fuzzy_topic_labels=fuzzy_topic_labels)
    
    # Plot fuzzy graphs for companies
    for permco, graph in company_fuzzy_graphs.items():
        plot_fuzzy_graph(graph, f'Fuzzy Graph for Company {permco}', fuzzy_topic_labels=fuzzy_topic_labels)
    
    # Plot average fuzzy graphs
    print("Creating and Plotting Average Fuzzy Graphs...")
    create_average_fuzzy_graphs(df, num_topics, top_clusters, fuzzy_topic_labels)
    
    print("Fuzzy Graph Generation and Plotting Completed.")
