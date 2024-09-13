import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os

# Load configuration from config.json
with open(config_hlr.json, 'r') as f:
    config = json.load(f)

# Get the correct model path from the config
model_load_path_with_data = config["model_load_path_with_data"]

# Load the embedding model to GPU
embedding_model = SentenceTransformer(config["embedding_model_choice"], device="cuda")

# Load the BERTopic model from the local path onto the GPU
topic_model = BERTopic.load(model_load_path_with_data, embedding_model=embedding_model)

# Directory to save outputs
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save basic info about the model
def save_basic_info(topic_model, output_file):
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

    print(f"Basic information saved to {output_file}")

# Save the reduced topics and info
def save_reduced_topics(topic_model, nr_topics, output_file):
    reduced_model = topic_model.reduce_topics(topic_model.original_documents_, nr_topics=nr_topics)
    save_basic_info(reduced_model, output_file)
    return reduced_model

# Save a plot of the topic distribution
def save_topics_distribution(topic_model, output_file):
    topic_info = topic_model.get_topic_info()
    plt.figure()
    plt.bar(topic_info['Topic'], topic_info['Count'])
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.title('Topic Distribution')
    plt.savefig(output_file)
    plt.close()
    print(f"Topic distribution saved to {output_file}")

# Save a visualization from BERTopic's built-in visualizations
def save_visualization(fig, output_file):
    fig.write_image(output_file)
    print(f"Visualization saved to {output_file}")

# Generate and save additional visualizations
def generate_additional_visualizations(topic_model):
    # Visualize Topics
    fig = topic_model.visualize_topics()
    save_visualization(fig, os.path.join(output_dir, "topics.html"))

    # Visualize Documents
    fig = topic_model.visualize_documents(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "documents.html"))

    # Visualize Document with DataMapPlot
    fig = topic_model.visualize_document_datamap(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "document_datamap.html"))

    # Visualize Document Hierarchy
    fig = topic_model.visualize_hierarchical_documents(topic_model.original_documents_)
    save_visualization(fig, os.path.join(output_dir, "hierarchical_documents.html"))

    # Visualize Topic Hierarchy
    fig = topic_model.visualize_hierarchy()
    save_visualization(fig, os.path.join(output_dir, "topic_hierarchy.html"))

    # Visualize Topic Terms (BarChart)
    fig = topic_model.visualize_barchart()
    save_visualization(fig, os.path.join(output_dir, "topic_barchart.html"))

    # Visualize Topic Similarity (Heatmap)
    fig = topic_model.visualize_heatmap()
    save_visualization(fig, os.path.join(output_dir, "topic_heatmap.html"))

    # Visualize Term Score Decline (Term Rank)
    fig = topic_model.visualize_term_rank()
    save_visualization(fig, os.path.join(output_dir, "term_rank.html"))

    # Visualize Topic Probability Distribution
    probabilities = topic_model.probabilities_  # Assuming we visualize the first document's probabilities
    if probabilities is not None:
        fig = topic_model.visualize_distribution(probabilities[0])
        save_visualization(fig, os.path.join(output_dir, "topic_distribution.html"))

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

print("All outputs saved.")
