import pandas as pd
import numpy as np
import os
import time
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
from text_splitting import extract_and_split_section  # Ensure split_text is not imported
from bertopic_training import load_bertopic_model

# variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10  # Adjusted for testing; number of unique companies to be analyzed, max is 1729
document_split = "paragraphs"  # Options are 'sentences', 'paragraphs', 'custom'; default is "paragraphs"
random_seed = 43  # Set a random seed for reproducibility
section_to_analyze = "Presentation"  # Can be "Presentation" or "Questions and Answers"; default is "Presentation"
max_documents = 30  # Limit the number of documents processed for testing
model_save_path = "D:/daten_masterarbeit/bertopic_model.pkl"  # Ensure consistent filename

# constants
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

def split_document(company, call_id, company_info, date, text, section_to_analyze, document_split):
    return extract_and_split_section(company, call_id, company_info, date, text, document_split, section_to_analyze)

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

def process_texts(permco, call_id, company_info, date, text, section_to_analyze, document_split):
    print(f"Starting analysis for company: {permco}, call ID: {call_id}")
    relevant_section = split_document(permco, call_id, company_info, date, text, section_to_analyze, document_split)
    if not relevant_section or len(relevant_section) == 0:
        print(f"Skipping company: {permco}, call ID: {call_id} due to missing or empty relevant section")
        return None
    return relevant_section

def main():
    start_time = time.time()
    # set random seed
    np.random.seed(random_seed)
    # Read the index file
    index_file = read_index_file(index_file_path)
    print("Index file loaded successfully.")

    # Compute and display descriptive statistics of index file
    compute_descriptive_statistics(index_file)

    # Create sample of earnings conference calls
    sample_start_time = time.time()
    ecc_sample = create_ecc_sample(sample_size, index_file, folderpath_ecc)
    sample_end_time = time.time()
    print(f"Sample creation took {sample_end_time - sample_start_time:.2f} seconds.")

    # Display the first 5 Companies of the sample (for demonstration)
    print("\nHere is the sample of earnings conference calls:")
    for i, (permco, calls) in enumerate(ecc_sample.items()):
        if i >= 5:
            break
        for key, value in calls.items():
            print(f"Permco_Key: {key}")
            print(f"Company Name: {value[0]}")
            print(f"Date: {value[1]}")
            print(f"Text Content: {value[2][1000:1100]}...")  # Displaying some letters from the Text.
            print()
    
    # Process each text to get the relevant sections
    all_relevant_sections = []
    result_dict = {}
    document_count = 0
    
    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            if document_count >= max_documents:
                break
            company_info, date, text = value
            relevant_sections = process_texts(permco, call_id, company_info, date, text, section_to_analyze, document_split)
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

    # Fit the BERTopic model once on all the relevant sections
    bertopic_start_time = time.time()
    if all_relevant_sections:
        print("Transforming documents with the BERTopic model...")
        topics, probabilities = topic_model.transform(all_relevant_sections)
        print(f"Topics generated: {topics}")
        print(f"Number of topics: {len(set(topics))}")
    else:
        print("No relevant sections found to transform with BERTopic.")
        return

    bertopic_end_time = time.time()
    print(f"BERTopic transformation took {bertopic_end_time - bertopic_start_time:.2f} seconds.")

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
            records.append({
                'permco': permco,
                'call_id': call_id,
                'company_info': company_info,
                'date': date,
                'text': sections,
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
    return results_df

if __name__ == "__main__":
    results_output = main()
