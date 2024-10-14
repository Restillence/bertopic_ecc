# fuzzy_graph.py
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import create_transition_matrix

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

def plot_fuzzy_graph(graph, call_id, fuzzy_topic_labels=None):
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
    
    plt.title(f'Fuzzy Graph for {call_id}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load configuration variables from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    input_path = config['topics_input_path']
    fuzzy_topic_labels = {int(k): v for k, v in config['fuzzy_topic_labels'].items()}

    # Create fuzzy graphs and optionally plot them
    df = pd.read_csv(input_path, sep='\t')
    num_topics = max(set([topic for sublist in df['filtered_topics'].apply(lambda x: eval(x)) for topic in sublist])) + 1
    fuzzy_graphs = create_fuzzy_graphs(df, num_topics)

    # Plot fuzzy graphs with the loaded fuzzy_topic_labels
    for call_id, graph in fuzzy_graphs.items():
        plot_fuzzy_graph(graph, call_id, fuzzy_topic_labels=fuzzy_topic_labels)
