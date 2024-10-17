# fuzzy_graph.py
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from utils import create_transition_matrix

# If git structure is not working properly:
fallback_config_path = "C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json"

def compute_similarity_to_average(df, num_topics):
    # Create transition matrices for each call
    transition_matrices = []
    call_ids = []
    siccds = []
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        transition_matrix = create_transition_matrix(topics, num_topics)
        transition_matrices.append(transition_matrix)
        call_ids.append(call_id)
        siccd = group['siccd'].iloc[0]  # Assuming 'siccd' is consistent within a call
        siccds.append(siccd)
    
    # Create a DataFrame to hold the data
    calls_df = pd.DataFrame({
        'call_id': call_ids,
        'transition_matrix': transition_matrices,
        'siccd': siccds
    })
    
    # Compute the overall average transition matrix
    average_transition_matrix = np.mean(transition_matrices, axis=0)
    average_transition_matrix = np.nan_to_num(average_transition_matrix)
    
    # Compute industry-specific average transition matrices
    industry_avg_matrices = {}
    for siccd, group in calls_df.groupby('siccd'):
        matrices = group['transition_matrix'].tolist()
        industry_avg_matrix = np.mean(matrices, axis=0)
        industry_avg_matrix = np.nan_to_num(industry_avg_matrix)
        industry_avg_matrices[siccd] = industry_avg_matrix
    
    # Compute similarities
    similarities_overall = []
    similarities_industry = []
    for idx, row in calls_df.iterrows():
        tm = np.nan_to_num(row['transition_matrix'])
        tm_vector = tm.flatten()
        # Similarity to overall average
        avg_vector = average_transition_matrix.flatten()
        sim_overall = 1 - cosine(tm_vector, avg_vector)
        similarities_overall.append(sim_overall)
        # Similarity to industry average
        industry_avg_matrix = industry_avg_matrices[row['siccd']]
        industry_avg_vector = industry_avg_matrix.flatten()
        sim_industry = 1 - cosine(tm_vector, industry_avg_vector)
        similarities_industry.append(sim_industry)
    
    # Add similarities to calls_df
    calls_df['similarity_to_overall_average'] = similarities_overall
    calls_df['similarity_to_industry_average'] = similarities_industry
    
    # Return the DataFrame with similarities
    similarity_df = calls_df[['call_id', 'similarity_to_overall_average', 'similarity_to_industry_average']]
    return similarity_df

def create_fuzzy_graphs(df, num_topics):
    graphs = {}
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        transition_matrix = create_transition_matrix(topics, num_topics)
        G = nx.DiGraph()
        for i in range(num_topics):
            G.add_node(i)
        for i in range(num_topics):
            for j in range(num_topics):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        graphs[call_id] = G
    return graphs

def create_industry_fuzzy_graphs(df, num_topics):
    # Collect transition matrices per industry
    industry_transition_matrices = {}
    grouped = df.groupby('siccd')
    for siccd, group in grouped:
        # Collect all topics in the industry
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        # Create transition matrix
        transition_matrix = create_transition_matrix(topics, num_topics)
        industry_transition_matrices[siccd] = transition_matrix
    
    # Create fuzzy graphs per industry
    industry_fuzzy_graphs = {}
    for siccd, transition_matrix in industry_transition_matrices.items():
        # Create graph
        G = nx.DiGraph()
        for i in range(num_topics):
            G.add_node(i)
        for i in range(num_topics):
            for j in range(num_topics):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        industry_fuzzy_graphs[siccd] = G
    return industry_fuzzy_graphs

def plot_fuzzy_graph(graph, title, fuzzy_topic_labels=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    node_labels = {node: f'Topic {node}' for node in graph.nodes()}
    
    if fuzzy_topic_labels:
        node_labels = {node: fuzzy_topic_labels.get(node, f'Topic {node}') for node in graph.nodes()}
    
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, width=[w * 5 for w in weights])
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
    except: 
        with open(fallback_config_path, 'r') as config_file:
            config = json.load(config_file)
            print("Config File Loaded.")

    input_path = config['merged_file_path']  # Updated to read the merged data with 'siccd'
    fuzzy_topic_labels = {int(k): v for k, v in config['fuzzy_topic_labels'].items()}

    # Read the merged data
    df = pd.read_csv(input_path)
    # Ensure 'filtered_topics' is evaluated as lists
    df['filtered_topics'] = df['filtered_topics'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    # Ensure 'siccd' is present
    if 'siccd' not in df.columns:
        raise ValueError("'siccd' column not found in the DataFrame.")

    # Compute num_topics
    num_topics = max(set([topic for sublist in df['filtered_topics'] for topic in sublist])) + 1

    # Compute similarities
    similarity_df = compute_similarity_to_average(df, num_topics)

    # Merge similarities back into df
    df = df.merge(similarity_df, on='call_id', how='left')

    # Create fuzzy graphs per call
    fuzzy_graphs = create_fuzzy_graphs(df, num_topics)

    # Create industry-specific fuzzy graphs
    industry_fuzzy_graphs = create_industry_fuzzy_graphs(df, num_topics)

    # Optionally, define topic labels
    topic_labels = fuzzy_topic_labels  # Using labels from config.json

    # Plot fuzzy graphs for a specific call (example)
    call_id_to_plot = df['call_id'].iloc[0]  # Replace with a specific call_id if desired
    graph_to_plot = fuzzy_graphs[call_id_to_plot]
    plot_fuzzy_graph(graph_to_plot, f'Fuzzy Graph for {call_id_to_plot}', fuzzy_topic_labels=topic_labels)

    # Plot fuzzy graphs for industries
    for siccd, graph in industry_fuzzy_graphs.items():
        plot_fuzzy_graph(graph, f'Fuzzy Graph for Industry {siccd}', fuzzy_topic_labels=topic_labels)
