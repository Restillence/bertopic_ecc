import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os
import time
import torch

# Load configuration from config.json
with open("config_hlr.json", 'r') as f:
    config = json.load(f)

# Get the correct model path from the config
model_load_path_with_data = os.path.abspath(config["model_load_path_with_data"])
print(f"Model load path with data: {model_load_path_with_data}")

# Start tracking time for loading the model
start_time = time.time()

# Load the embedding model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the embedding model to GPU
embedding_model = SentenceTransformer(config["embedding_model_choice"], device=device)

# Load the BERTopic model from the local path onto the GPU
topic_model = BERTopic.load(model_load_path_with_data, embedding_model=embedding_model)

# End tracking time for loading the model
end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds")

# Directory to save outputs
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save basic info about the model
def save_basic_info(topic_model, output_file):
    start_time = time.time()
    print("Saving basic information...")
    with open(output_file, 'w') as f:
        # Get the number of topics
        num_topics = len(topic_model.get_topic_info())
        f.write(f"Number of Topics: {num_topics}\n")
        
        # Get topic frequency (number of documents per topic)
        topic_info = topic_model.get_topic_info()
        f.write("Top 5 Topics by Frequency:\n")
        f.write(topic_info.head(5).to_string())
        f.write("\n\n")

        # Get top words for a specific topic (example: topic 0)
        example_topic = 0
        top_words = topic_model.get_topic(example_topic)
        f.write(f"Top words for Topic {example_topic}:\n")
        for word, score in top_words:
            f.write(f"  - {word}: {score}\n")

    end_time = time.time()
    print(f"Basic information saved to {output_file} in {end_time - start_time:.2f} seconds")

# Save the reduced topics and info
def save_reduced_topics(topic_model, nr_topics, output_file):
    start_time = time.time()
    print("Reducing topics...")
    reduced_model = topic_model.reduce_topics(topic_model.original_documents_, nr_topics=nr_topics)
    save_basic_info(reduced_model, output_file)
    end_time = time.time()
    print(f"Topics reduced to {nr_topics} and info saved in {end_time - start_time:.2f} seconds")
    return reduced_model

# Save a plot of the topic distribution
def save_topics_distribution(topic_model, output_file):
    start_time = time.time()
    print("Saving topic distribution...")
    topic_info = topic_model.get_topic_info()
    plt.figure()
    plt.bar(topic_info['Topic'], topic_info['Count'])
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.title('Topic Distribution')
    plt.savefig(output_file)
    plt.close()
    end_time = time.time()
    print(f"Topic distribution saved to {output_file} in {end_time - start_time:.2f} seconds")

# Save a visualization from BERTopic's built-in visualizations
def save_visualization(fig, output_file, file_format="png"):
    start_time = time.time()
    print(f"Saving visualization to {output_file}...")
    if file_format == "html":
        fig.write_html(output_file)
    else:
        fig.write_image(output_file)
    end_time = time.time()
    print(f"Visualization saved to {output_file} in {end_time - start_time:.2f} seconds")

# Generate and save additional visualizations
def generate_additional_visualizations(topic_model):
    print("Generating additional visualizations...")
    
    start_time = time.time()
    
    # Visualize Topics
    print("Visualizing topics...")
    fig = topic_model.visualize_topics()
    save_visualization(fig, os.path.join(output_dir, "topics.html"), file_format="html")

    # Visualize Documents    #takes very long atm!!!
    print("Visualizing documents...")
    fig = topic_model.visualize_documents(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "documents.html"), file_format="html")

    # Visualize Document with DataMapPlot
    print("Visualizing document with datamapplot...")
    fig = topic_model.visualize_document_datamap(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "document_datamap.html"), file_format="html")

    # Visualize Document Hierarchy
    print("Visualizing document hierarchy...")
    fig = topic_model.visualize_hierarchical_documents(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "hierarchical_documents.html"), file_format="html")

    # Visualize Topic Hierarchy
    print("Visualizing topic hierarchy...")
    fig = topic_model.visualize_hierarchy()
    save_visualization(fig, os.path.join(output_dir, "topic_hierarchy.html"), file_format="html")

    # Visualize Topic Terms (BarChart)
    print("Visualizing topic terms...")
    fig = topic_model.visualize_barchart()
    save_visualization(fig, os.path.join(output_dir, "topic_barchart.html"), file_format="html")

    # Visualize Topic Similarity (Heatmap)
    print("Visualizing topic similarity...")
    fig = topic_model.visualize_heatmap()
    save_visualization(fig, os.path.join(output_dir, "topic_heatmap.html"), file_format="html")

    # Visualize Term Score Decline (Term Rank)
    print("Visualizing term rank...")
    fig = topic_model.visualize_term_rank()
    save_visualization(fig, os.path.join(output_dir, "term_rank.html"), file_format="html")

    # Visualize Topic Probability Distribution
    print("Visualizing topic distribution...")
    probabilities = topic_model.probabilities_  # Assuming we visualize the first document's probabilities
    if probabilities is not None:
        fig = topic_model.visualize_distribution(probabilities[0])
        save_visualization(fig, os.path.join(output_dir, "topic_distribution.html"), file_format="html")

    end_time = time.time()
    print(f"All visualizations generated and saved in {end_time - start_time:.2f} seconds")

# Start tracking overall script time
script_start_time = time.time()

# Save basic info about the model
save_basic_info(topic_model, os.path.join(output_dir, "basic_info.txt"))

# Save a plot of the topic distribution
save_topics_distribution(topic_model, os.path.join(output_dir, "topic_distribution.png"))

# Reduce topics to 30 and save new info
reduced_model = save_reduced_topics(topic_model, nr_topics=30, output_file=os.path.join(output_dir, "reduced_info.txt"))

# Save the reduced topic distribution
save_topics_distribution(reduced_model, os.path.join(output_dir, "reduced_topic_distribution.png"))

# Generate and save the additional visualizations
generate_additional_visualizations(topic_model)

# End tracking overall script time
script_end_time = time.time()
print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")
