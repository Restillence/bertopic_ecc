# filehandling.py
import os
import pandas as pd
import numpy as np

class FileHandler:
    """
    Class to handle file operations

    Attributes
    ----------
    config : dict
        Configuration dictionary containing file paths and settings
    """

    def __init__(self, config):
        """
        Initializes the FileHandler

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        # Set the configuration
        self.config = config

        # Extract file paths from config
        self.index_file_path = self.config.get('index_file_path')
        self.folderpath_ecc = self.config.get('folderpath_ecc')

        # Ensure file paths are set
        if not self.index_file_path or not self.folderpath_ecc:
            raise ValueError("File paths 'index_file_ecc_folder' and 'folderpath_ecc' must be set in the config.")

    def read_index_file(self):
        """
        Reads the index file into a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The index file as a pandas DataFrame
        """
        # Read the index file
        index_file = pd.read_csv(self.index_file_path, sep=";")
        return index_file

    def create_ecc_sample(self, sample_size):
        """
        Creates a sample of earnings call transcripts based on the sampling mode specified in the config.

        Parameters
        ----------
        sample_size : int
            The number of items to include in the sample

        Returns
        -------
        dict
            A dictionary containing the sampled earnings call transcripts
        """
        # Read the index file
        index_file = self.read_index_file()
        # Initialize the ECC sample
        ecc_sample = {}

        # Get all the files in the ECC folder
        all_files = os.listdir(self.folderpath_ecc)
        print("First 10 files in directory:", all_files[:10])

        # Get the sampling mode from the config, default to 'random_company'
        sampling_mode = self.config.get('sampling_mode', 'random_company')

        if sampling_mode == 'full_random':
            # Sample earnings calls completely at random
            print("Sampling mode: full_random")
            # Limit sample size to the number of available files
            sample_size = min(sample_size, len(all_files))
            # Select random files
            random_files = np.random.choice(all_files, size=sample_size, replace=False)
            print(f"Selected {len(random_files)} random files.")
            print("First 10 random files:", random_files[:10])
            for ecc_file in random_files:
                # Extract permco and SE_ID from the file name
                parts = ecc_file.replace('.txt', '').split('_')
                if len(parts) != 4:
                    print(f"File name {ecc_file} does not match expected pattern.")
                    continue
                if parts[0] != 'earnings' or parts[1] != 'call':
                    print(f"File name {ecc_file} does not match expected pattern.")
                    continue
                permco_str = parts[2]
                se_id_str = parts[3]
                try:
                    permco = int(permco_str)
                    se_id = int(se_id_str)
                except ValueError:
                    print(f"Cannot convert permco or SE_ID to int in file name {ecc_file}.")
                    continue

                # Define ecc_key here
                ecc_key = f"earnings_call_{permco}_{se_id}"

                # Get the row for the current permco and SE_ID
                specific_row = index_file[(index_file['permco'] == permco) & (index_file['SE_ID'] == se_id)]
                if not specific_row.empty:
                    company_name = specific_row.iloc[0]['company_name_TR']
                    date = specific_row.iloc[0]['date']
                else:
                    company_name = 'Unknown'
                    date = 'Unknown'
                    print(f"SE_ID {se_id} with permco {permco} not found in index file.")

                # Open the file and read the text content
                with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                    text_content = file.read()

                # Add the ECC to the sample
                if permco not in ecc_sample:
                    ecc_sample[permco] = {}
                ecc_sample[permco][ecc_key] = {
                    'company_name': company_name,
                    'date': date,
                    'text_content': text_content
                }

        elif sampling_mode == 'random_company':
            # Sample companies at random and include their earnings calls
            print("Sampling mode: random_company")
            # Get the unique companies
            unique_companies = index_file['permco'].unique()
            # Limit the sample size to the number of unique companies
            sample_size = min(sample_size, len(unique_companies))
            # Select a random subset of the unique companies
            random_companies = np.random.choice(unique_companies, size=sample_size, replace=False)

            print("Starting to create ECC sample...")
            for permco in random_companies:
                # Get the rows for the current permco
                company_rows = index_file[index_file['permco'] == permco]
                # Get the company name
                company_name = company_rows.iloc[0]['company_name_TR']

                # Get the ECC files for the current permco
                ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]

                print(f"Found {len(ecc_files)} files for permco {permco}")

                for ecc_file in ecc_files:
                    # Get the SE ID from the file name
                    se_id = ecc_file.split('_')[3].replace('.txt', '')
                    # Create the ECC key
                    ecc_key = f"earnings_call_{permco}_{se_id}"

                    # Get the row for the current SE ID
                    specific_row = company_rows[company_rows['SE_ID'] == int(se_id)]
                    # Get the date from the row
                    if not specific_row.empty:
                        date = specific_row.iloc[0]['date']
                    else:
                        date = 'Unknown'
                        print(f"SE_ID {se_id} not found in index file for permco {permco}")

                    # Open the file and read the text content
                    with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                        text_content = file.read()

                    # Add the ECC to the sample
                    if permco not in ecc_sample:
                        ecc_sample[permco] = {}
                    ecc_sample[permco][ecc_key] = {
                        'company_name': company_name,
                        'date': date,
                        'text_content': text_content
                    }
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        return ecc_sample
