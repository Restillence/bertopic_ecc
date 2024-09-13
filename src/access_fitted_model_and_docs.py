import json
from bertopic import BERTopic

# Load the configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Get the correct model path from the updated config
model_load_path_with_data = config["model_load_path_with_data"]

# Load the model from the local path
loaded_topic_model = BERTopic.load(model_load_path_with_data)

# Access the original documents (untransformed)
original_documents = loaded_topic_model.original_documents_

# Access the transformed topics and probabilities
transformed_topics = loaded_topic_model.topics_
transformed_probabilities = loaded_topic_model.probabilities_

# Print document, topic assignment, probability, and topic representation
for i, doc in enumerate(original_documents):
    assigned_topic = transformed_topics[i]
    topic_probability = transformed_probabilities[i]
    
    # Print document info
    print(f"Document: {doc}")
    print(f"Assigned Topic: {assigned_topic}")
    print(f"Topic Probability: {topic_probability}")
    
    # If the assigned topic is -1, it means no topic was assigned
    if assigned_topic != -1:
        # Get the topic representation (top words for the topic)
        topic_representation = loaded_topic_model.get_topic(assigned_topic)
        print(f"Topic Representation for Topic {assigned_topic}:")
        for word, score in topic_representation:
            print(f"  - {word}: {score}")
    else:
        print("No topic was assigned to this document.")
    
    print("\n" + "-"*50 + "\n")  # Separator for better readability
