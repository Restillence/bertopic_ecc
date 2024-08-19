#filehandling.py
import re
import pandas as pd
import numpy as np
import os

def clean_csv_file(filepath):
    cleaned_file_path = filepath.replace(".csv", "_cleaned.csv")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df.to_csv(cleaned_file_path, index=False)
        return cleaned_file_path
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None

def read_index_file(index_file_path):
    index_file = pd.read_csv(index_file_path, sep=";")
    return index_file

def create_ecc_sample(sample_size, index_file, folderpath_ecc):
    ecc_sample = {}

    # Ensure sample size does not exceed number of unique companies (permco)
    unique_companies = index_file['permco'].unique()
    sample_size = min(sample_size, len(unique_companies))
    random_companies = np.random.choice(unique_companies, size=sample_size, replace=False)
    # random_companies = [37,82,116,176,211]  # For debugging; remove or comment out for actual random sampling

    all_files = os.listdir(folderpath_ecc)
    print("All files in directory:", all_files[:10])  # Print the first 10 files for debugging
    
    print("Starting to create ECC sample...")
    for permco in random_companies:
        # Get company details from index file
        company_rows = index_file[index_file['permco'] == permco]
        company_name = company_rows.iloc[0]['company_name_TR']
        
        #print(f"Processing company: {company_name} with permco: {permco}")  # Debugging line

        # Find all ECC files for this permco
        ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]
        
        print(f"Found {len(ecc_files)} files for permco {permco}")  # Debugging line

        for ecc_file in ecc_files:
            se_id = ecc_file.split('_')[3].replace('.txt', '')  # Extract SE_ID from filename
            ecc_key = f"earnings_call_{permco}_{se_id}"

            #print(f"Processing file: {ecc_file}, SE_ID: {se_id}")  # Debugging line

            # Get the specific date for this earnings call
            specific_row = company_rows[company_rows['SE_ID'] == int(se_id)]
            if not specific_row.empty:
                date = specific_row.iloc[0]['date']
            else:
                date = 'Unknown'  # In case the SE_ID is not found
                print(f"SE_ID {se_id} not found in index file for permco {permco}")  # Debugging line

            # Read the text content of the ECC file
            with open(os.path.join(folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                text_content = file.read()

            # Construct the dictionary entry
            if permco not in ecc_sample:
                ecc_sample[permco] = {}
            ecc_sample[permco][ecc_key] = (company_name, date, text_content)

    return ecc_sample