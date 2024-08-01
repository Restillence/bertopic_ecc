"""
This file tests the text splitting functionality
"""

# imports
import pandas as pd
import numpy as np
import os
import time
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
from text_splitting import extract_and_split_section

# variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 2  # number of unique companies to be analyzed, max is 1729
document_split = "sentences"  # Options are 'sentences', 'paragraphs', 'custom'; default is "paragraphs"
random_seed = 42  # Set a random seed for reproducibility
section_to_analyze = "Questions and Answers" # Can be "Presentation" or "Questions and Answers"; default is "Presentation"

# constants
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

def main():
    start_time = time.time()
    np.random.seed(random_seed)

    # Read the index file
    index_file = read_index_file(index_file_path)
    print("Index file loaded successfully.")

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

    for permco, calls in ecc_sample.items():
        for call_id, details in calls.items():
            company = details[0]
            date = details[1]
            text = details[2]
            split_section = extract_and_split_section(company, call_id, details, date, text, document_split, section_to_analyze)
            if split_section is None:
                print(f"Skipping company: {company}, call ID: {call_id} due to missing relevant section")
                continue  # Skip if the relevant section is not found
            result_dict[(permco, call_id)] = {
                "company": company,
                "date": date,
                "split_texts": split_section
            }
            print(f"Processed company: {company}, call ID: {call_id}")

    # Save the results to a file
    if result_dict:
        results_df = pd.DataFrame.from_dict(result_dict, orient='index')
        results_df.to_csv(index_file_ecc_folder + "split_texts_results.csv")
        print("Results saved to split_texts_results.csv")
    else:
        print("No results to save.")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    return result_dict

if __name__ == "__main__":
    test_output = main()
