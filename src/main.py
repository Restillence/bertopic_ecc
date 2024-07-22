"""
This file contains the main functionality of the program.
I use BERTopic to analyze ECC data.
"""

#imports
import pandas as pd
import numpy as np
import os
import time
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
from text_splitting import split_text

#variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 2  # number of unique companies to be analyzed, max is 1729
document_split = "sentences" #TODO right now it is only working for 'sentences', 'paragraphs' is not possible, fix it!

#constants
#nothing to change here
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

def split_document(): #this function might go to a submodule
    pass

def build_data_pipeline_ecc():   #this function might go to a submodule
    pass

def match_ecc_financial_data(): #this function might go to a submodule
    pass

def compute_descriptive_statistics(df):
    if df is not None:
        print("Here are some Descriptive Statistics of the index file:")
        print(df.head(5))
        print(df.columns)
        print("Number of unique companies:")
        print(df['permco'].nunique())  # Number of unique companies
        print("other descriptive statistics:")
        print(df.describe())
    else:
        print("Failed to load index file.")


def process_texts(topic_model, company, call_id, company_info, date, text):
    print(f"Splitting text for company: {company}, call ID: {call_id}")
    split_texts = split_text(text, document_split)
    topics, probabilities = topic_model.fit_transform(split_texts)
    print("Fitting model...")
    return topics

def main():
    start_time = time.time()
    
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
    

    
    # Process each text with BERTopic
    # Initialize the result dictionary
    result_dict = {}

    # Initialize BERTopic with KeyBERTInspired representation
    topic_model = BERTopic(representation_model=KeyBERTInspired())
    
    # Process each text with BERTopic for only the first 5 calls 
    bertopic_start_time = time.time()
    processed_calls = 0  # Counter to keep track of the number of processed calls #TODO this has to be removed!!!
    
    for permco, calls in ecc_sample.items():
        for call_id, value in calls.items():
            if processed_calls >= 3:
                break
            company_info, date, text = value
            topics = process_texts(topic_model, permco, call_id, company_info, date, text)
            if permco not in result_dict:
                result_dict[permco] = {}
            result_dict[permco][call_id] = (company_info, date, text, topics)
            processed_calls += 1
    
        if processed_calls >= 3:
            break
    
    bertopic_end_time = time.time()
    print(f"BERTopic fitting took {bertopic_end_time - bertopic_start_time:.2f} seconds.")

    # Convert the results to a DataFrame and save it
    records = []
    for permco, calls in result_dict.items():
        for call_id, values in calls.items():
            company_info, date, text, topics = values
            records.append({
                'permco': permco,
                'call_id': call_id,
                'company_info': company_info,
                'date': date,
                'text': text,
                'topics': topics
            })
    
    # Save the results
    results_df = pd.DataFrame(records)
    results_output_path = os.path.join(index_file_ecc_folder, 'topics_output.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")
    
    # Visualize topics
    topic_model.visualize_topics().show()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
