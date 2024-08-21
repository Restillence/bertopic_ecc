import pandas as pd
import numpy as np
import os
import time
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
import json
from utils import print_configuration, load_bertopic_model

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract variables from the config
folderpath_ecc = config["folderpath_ecc"]
index_file_ecc_folder = config["index_file_ecc_folder"]
sample_size = config["sample_size"]
document_split = config["document_split"]
random_seed = config["random_seed"]
section_to_analyze = config["section_to_analyze"]
max_documents = config["max_documents"]
model_save_path = config["model_save_path"]
index_file_path = config["index_file_path"]
embeddings_path = config["embeddings_path"]

print_configuration(config)

# Initialize FileHandler and TextProcessor
file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

def compute_descriptive_statistics(df):
    if df is not None:
        print("Here are some Descriptive Statistics of the index file:")
        print(df.head(5))
        print(df.columns)
        print("Number of unique companies:")
        print(df['permco'].nunique())  # Number of unique companies
        print("Other descriptive statistics:")
        print(df.describe())
    else:
        print("Failed to load index file.")

def process_texts(permco, call_id, company_info, date, text):
    print(f"Starting analysis for company: {permco}, call ID: {call_id}")
    relevant_section = text_processor.extract_and_split_section(permco, call_id, company_info, date, text)
    if not relevant_section or len(relevant_section) == 0:
        print(f"Skipping company: {permco}, call ID: {call_id} due to missing or empty relevant section")
        return None
    return relevant_section

def main():
    start_time = time.time()
    np.random.seed(random_seed)  # Set random seed

    # Read the index file
    index_file = file_handler.read_index_file()
    print("Index file loaded successfully.")
    
    # Compute and display descriptive statistics of index file
    compute_descriptive_statistics(index_file)

    # Create a sample of earnings conference calls
    sample_start_time = time.time()
    ecc_sample = file_handler.create_ecc_sample(sample_size)
    sample_end_time = time.time()
    print(f"Sample creation took {sample_end_time - sample_start_time:.2f} seconds.")

    # Process each text to get the relevant sections
    all_relevant_sections = []
    result_dict = {}
    document_count = 0

    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            if document_count >= max_documents:
                break
            company_info, date, text = value
            relevant_sections = process_texts(permco, call_id, company_info, date, text)
            if relevant_sections is not None:
                if isinstance(relevant_sections, list):
                    all_relevant_sections.extend(relevant_sections)
                else:
                    relevant_sections = [relevant_sections]
                    all_relevant_sections.extend(relevant_sections)
                if permco not in result_dict:
                    result_dict[permco] = {}
                if call_id not in result_dict[permco]:
                    result_dict[permco][call_id] = (company_info, date, text, relevant_sections)
                else:
                    result_dict[permco][call_id][-1].extend(relevant_sections)
                document_count += 1
        if document_count >= max_documents:
            break

    # Ensure all elements in all_relevant_sections are strings
    all_relevant_sections = [str(section) for section in all_relevant_sections]

    # Load the trained BERTopic model
    topic_model = load_bertopic_model(model_save_path)

    # Load the embeddings
    embeddings = np.load(embeddings_path)

    # Transform the documents with the BERTopic model using precomputed embeddings
    bertopic_start_time = time.time()
    if all_relevant_sections:
        print("Transforming documents with the BERTopic model...")
        topics, probabilities = topic_model.transform(all_relevant_sections, embeddings=embeddings)
        end_time = time.time()
        print(f"BERTopic model transformed {len(all_relevant_sections)} sections.")
        print(f"Transformation time: {end_time - bertopic_start_time:.2f} seconds.")
    else:
        print("No relevant sections found to transform with BERTopic.")
        return

    # Update the result_dict with topics for each section
    topic_idx = 0
    for permco, calls in result_dict.items():
        for call_id in calls.keys():
            if topic_idx < len(topics):
                num_sections = len(result_dict[permco][call_id][-1])
                section_topics = topics[topic_idx:topic_idx + num_sections]
                result_dict[permco][call_id] = (*result_dict[permco][call_id], section_topics)
                topic_idx += num_sections
            else:
                result_dict[permco][call_id] = (*result_dict[permco][call_id], ['No topics found'])

    # Convert the results to a DataFrame and save it
    records = []
    for permco, calls in result_dict.items():
        for call_id, values in calls.items():
            if len(values) == 5:
                company_info, date, text, sections, section_topics = values
            else:
                company_info, date, text, sections = values
                section_topics = ['No topics found'] * len(sections)
            
            # Ensure 'sections' is a list of strings
            if not isinstance(sections, list) or not all(isinstance(section, str) for section in sections):
                raise ValueError(f"Sections should be a list of strings, but got {type(sections)} with elements {type(sections[0]) if sections else 'N/A'}")
            
            # Assign the sections directly to text_list
            text_list = sections
            
            records.append({
                'permco': permco,
                'call_id': call_id,
                'company_info': company_info,
                'date': date,
                'text': text_list,  # Assign the list of strings directly
                'topics': section_topics
            })

    # Save the results
    results_df = pd.DataFrame(records)
    results_output_path = os.path.join(index_file_ecc_folder, 'topics_output.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")
    
    # Visualize topics
    try:
        if len(all_relevant_sections) > 0 and len(topics) > 0:
            print("Visualizing topics...")
            topic_model.visualize_topics().show()
        else:
            print("No topics to visualize.")
    except ValueError as ve:
        print(f"Visualization error: {ve}")

    # Get detailed topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopic Info:")
    print(topic_info)

    # Display terms for each topic
    for topic in topic_info['Topic'].unique():
        if topic != -1:  # -1 is typically the outlier/noise topic
            print(f"\nTopic {topic}:")
            for word, score in topic_model.get_topic(topic):
                print(f"{word}: {score:.4f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
