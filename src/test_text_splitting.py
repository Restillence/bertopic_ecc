"""
This file tests the text splitting functionality
"""

# imports
import pandas as pd
import numpy as np
import os
import time
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
from text_splitting import split_text, process_texts, split_text_by_visual_cues

# variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 2  # number of unique companies to be analyzed, max is 1729
document_split = "paragraphs"  # TODO right now it is only working for 'sentences', 'paragraphs' is not possible, fix it!
random_seed = 42  # Set a random seed for reproducibility
#section_to_analyze = "Presentation" #can be "Presentation" or "Q&A"; right now still hardcoded, should be changed later

# constants
# nothing to change here
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

"""
def process_texts(company, call_id, company_info, date, text):
    print(f"Splitting text for company: {company}, call ID: {call_id}")
    split_texts = split_text(text, document_split)
    return split_texts
"""

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
            extract_relevant_section = process_texts(company, call_id, details, date, text, document_split) #right now extracts the "Presentation" section
            split_section = split_text_by_visual_cues(extract_relevant_section)
            result_dict[(permco, call_id)] = {
                "company": company,
                "date": date,
                "split_texts": split_section
            }
    
    # Save the results to a file
    results_df = pd.DataFrame.from_dict(result_dict, orient='index')
    results_df.to_csv(index_file_ecc_folder + "split_texts_results.csv")
    print("Results saved to split_texts_results.csv")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    return result_dict

if __name__ == "__main__":
    test_output = main()

