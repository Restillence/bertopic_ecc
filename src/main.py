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

#variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10  # number of unique companies to be analyzed, max is 1729
document_split = "sentences"

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
        print("Here are some Descriptive Statistics:")
        print(df.head(5))
        print(df.columns)
        print("Number of unique companies:")
        print(df['permco'].nunique())  # Number of unique companies
        print("other descriptive statistics:")
        print(df.describe())
    else:
        print("Failed to load index file.")

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
    for i, (key, value) in enumerate(ecc_sample.items()):
        if i >= 5:
            break
        print(f"Key: {key}")
        print(f"Company Info: {value[0]}")
        print(f"Text Content: {value[1][:100]}...")  # Displaying first 100 characters of text
        print()

    # Extract texts for BERTopic analysis
    texts = [value[1] for value in ecc_sample.values()]
    
    # Initialize BERTopic with KeyBERTInspired representation
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(representation_model=representation_model)

    # Fit the model on the texts
    bertopic_start_time = time.time()
    topics, probabilities = topic_model.fit_transform(texts)
    bertopic_end_time = time.time()
    print(f"BERTopic fitting took {bertopic_end_time - bertopic_start_time:.2f} seconds.")

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'key': list(ecc_sample.keys()),
        'company_info': [value[0] for value in ecc_sample.values()],
        'text': texts,
        'topic': topics
    })

    # Save the results
    results_output_path = os.path.join(index_file_ecc_folder, 'topics_output.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")

    # Visualize topics
    topic_model.visualize_topics().show()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
