"""
This file contains the main functionality of the program.
I use BERTopic to analyze ECC data.
"""

#imports
import pandas as pd
import numpy as np
import os
import time
import json
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
from text_splitting import split_text  # Import the text_splitting module
from multiprocessing import Pool, cpu_count  # Added missing imports

#variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 1  # number of unique companies to be analyzed, max is 1729
document_split = "sentences"  # Choose between 'sentences', 'paragraphs', or 'custom'

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

def process_texts(data):
    company, call_id, company_info, date, text = data
    print(f"Splitting text for company: {company}, call ID: {call_id}")
    split_texts = split_text(text, document_split)
    topic_model = BERTopic(representation_model=KeyBERTInspired())
    topics, probabilities = topic_model.fit_transform(split_texts)
    return (company, call_id, topics)

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

    # Display the first 5 items of the sample (for demonstration)
    print("\nHere is the sample of earnings conference calls:")
    for i, (permco, calls) in enumerate(ecc_sample.items()):
        if i >= 5:
            break
        for key, value in calls.items():
            print(f"Permco_Key: {key}")
            print(f"Company Name: {value[0]}")
            print(f"Text Content: {value[2][:100]}...")  # Displaying first 100 characters of text
            print()

    # Prepare data for multiprocessing
    print("Preparing data for multiprocessing...")
    data = [(str(company), call_id, company_info, date, text) for company, calls in ecc_sample.items() for call_id, (company_info, date, text) in calls.items()]

    # Analyze the texts with BERTopic using multiprocessing
    bertopic_start_time = time.time()
    with Pool(cpu_count()) as pool:
        results = pool.map(process_texts, data)
    bertopic_end_time = time.time()
    print(f"BERTopic fitting took {bertopic_end_time - bertopic_start_time:.2f} seconds.")

    # Collect results into result_dict
    result_dict = {}
    for company, call_id, topics in results:
        if company not in result_dict:
            result_dict[company] = {}
        result_dict[company][str(call_id)] = topics

    # Save the results
    results_output_path = os.path.join(index_file_ecc_folder, 'topics_output.json')
    with open(results_output_path, 'w') as outfile:
        json.dump(result_dict, outfile)
    print(f"Results saved to {results_output_path}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    return result_dict

if __name__ == "__main__":
    results_dict = main()
