import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Load your data
df_new = pd.read_csv(
    "D:/daten_masterarbeit/topics_output_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv",
    sep='\t'
)
df_new['filtered_topics'] = df_new['filtered_topics'].apply(lambda x: eval(x))

# Extract all unique topic indices
all_topics = [topic for sublist in df_new['filtered_topics'] for topic in sublist]
unique_topics = set(all_topics)
num_topics = max(unique_topics) + 1

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

def create_fuzzy_graphs(df, num_topics):
    graphs = {}
    grouped = df.groupby('call_id')
    for call_id, group in grouped:
        # Flatten the list of topics for each call
        topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
        # Create transition matrix
        transition_matrix = create_transition_matrix(topics, num_topics)
        # Create graph
        G = nx.DiGraph()
        for i in range(num_topics):
            G.add_node(i)
        # Add edges with weights from the transition matrix
        for i in range(num_topics):
            for j in range(num_topics):
                weight = transition_matrix[i][j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        graphs[call_id] = G
    return graphs

def plot_fuzzy_graph(graph, call_id, topic_labels=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    node_labels = {node: f'Topic {node}' for node in graph.nodes()}
    
    if topic_labels:
        node_labels = {node: topic_labels.get(node, f'Topic {node}') for node in graph.nodes()}
    
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, width=[w * 5 for w in weights])
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title(f'Fuzzy Graph for {call_id}')
    plt.axis('off')
    plt.show()

# Create transition matrices and collect them
transition_matrices = []
call_ids = []
grouped = df_new.groupby('call_id')
for call_id, group in grouped:
    topics = [topic for sublist in group['filtered_topics'] for topic in sublist]
    transition_matrix = create_transition_matrix(topics, num_topics)
    transition_matrices.append(transition_matrix)
    call_ids.append(call_id)

# Compute the average transition matrix
average_transition_matrix = np.mean(transition_matrices, axis=0)
average_transition_matrix = np.nan_to_num(average_transition_matrix)

# Compute similarity between each call's transition matrix and the average using cosine similarity
similarities = []
for tm in transition_matrices:
    tm = np.nan_to_num(tm)
    tm_vector = tm.flatten()
    avg_vector = average_transition_matrix.flatten()
    sim = 1 - cosine(tm_vector, avg_vector)
    similarities.append(sim)

# Create a DataFrame with call_ids and similarities
similarity_df = pd.DataFrame({'call_id': call_ids, 'similarity_to_average': similarities})

# Merge the similarity DataFrame back into df_new
df_new = df_new.merge(similarity_df, on='call_id', how='left')
df_new = df_new[["permco","call_id","company_info","date","filtered_topics",
                 "filtered_texts", "similarity_to_average"]]

# Save the DataFrame to a CSV file
df_new.to_csv("topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv", index=False)

# Create fuzzy graphs
fuzzy_graphs = create_fuzzy_graphs(df_new, num_topics)

#TODO, Achtung, Topic Labels überprüfen!!!
# Define topic labels if you have them
topic_labels = {
    0: 'Introduction',
    1: 'Revenue',
    2: 'Expenses',
    3: 'Earnings',
    4: 'Marketing',
    5: 'Strategy',
    6: 'Risk'
    # Add labels for additional topics if any
}

#optional: plotting
"""
# Choose a call_id to plot
call_id_to_plot = 'earnings_call_16473_10443834'  # Replace with a call_id from your data

# Get the graph
graph_to_plot = fuzzy_graphs[call_id_to_plot]

# Plot the 
for call_id, graph in fuzzy_graphs.items():
    plot_fuzzy_graph(graph, call_id, topic_labels=topic_labels)
#plot_fuzzy_graph(graph_to_plot, call_id_to_plot, topic_labels=topic_labels)
"""