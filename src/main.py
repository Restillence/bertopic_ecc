"""
This file contains the main functionality of the program.
I use BERTtopic to analyze ECC data.
"""

#imports
import pandas as pd
import numpy as np
import os
from file_handling import read_index_file, create_ecc_sample # Import the file_handling module

#variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"   
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10 # number of unique companies to be analyzed, max is 1729

#constants
#nothing to change here
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

def locate_file():
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
    # Read the index file
    index_file = read_index_file(index_file_path)
    print("index file loaded successfully.")

    # Compute and display descriptive statistics of index file
    compute_descriptive_statistics(index_file)

    # Create sample of earnings conference calls
    ecc_sample = create_ecc_sample(sample_size, index_file, folderpath_ecc)
    
    # Display the sample (for demonstration)
    print("\nHere is the sample of earnings conference calls:")
    for i, (key, value) in enumerate(ecc_sample.items()):
        if i >= 5:
            break
        print(f"Key: {key}")
        print(f"Company Info: {value[0]}")
        print(f"Text Content: {value[1][:100]}...")  # Displaying first 100 characters of text
        print()



if __name__ == "__main__":
    main()
