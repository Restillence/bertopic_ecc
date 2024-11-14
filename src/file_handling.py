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
        Creates a sample of earnings call transcripts based on the sampling mode specified in the config,
        excluding calls with fewer than 1600 words.

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
        excluded_count = 0  # Counter for excluded calls

        # Get all the files in the ECC folder
        all_files = os.listdir(self.folderpath_ecc)
        print("First 10 files in directory:", all_files[:10])

        # Regular expression pattern to match the expected filename format
        pattern = re.compile(r'^earnings_call_(\d+)_(\d+)\.txt$')

        # Get the sampling mode from the config, default to 'random_company'
        sampling_mode = self.config.get('sampling_mode', 'random_company')

        if sampling_mode == 'full_random':
            # Sample earnings calls completely at random
            print("Sampling mode: full_random")
            while len(ecc_sample) < sample_size:
                remaining_sample_size = sample_size - len(ecc_sample)
                selected_files = np.random.choice(all_files, size=remaining_sample_size, replace=False)
                
                for ecc_file in selected_files:
                    if len(ecc_sample) >= sample_size:
                        break

                    # Match filename against the pattern
                    match = pattern.match(ecc_file)
                    if not match:
                        print(f"File name {ecc_file} does not match expected pattern. Skipping file.")
                        continue
                    
                    # Extract permco and SE_ID from the match groups
                    permco, se_id = int(match.group(1)), int(match.group(2))
                    ecc_key = f"earnings_call_{permco}_{se_id}"

                    # Get company information from the index file
                    specific_row = index_file[(index_file['permco'] == permco) & (index_file['SE_ID'] == se_id)]
                    company_name = specific_row.iloc[0]['company_name_TR'] if not specific_row.empty else 'Unknown'
                    date = specific_row.iloc[0]['date'] if not specific_row.empty else 'Unknown'

                    # Open and read text content
                    with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                        text_content = file.read()

                    # Check if the word count is above the threshold (1600 words)
                    word_count = len(text_content.split())
                    if word_count < 1600:
                        excluded_count += 1
                        continue  # Skip this file if below the word count threshold

                    # Add to sample if word count is sufficient
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
            unique_companies = index_file['permco'].unique()

            while len(ecc_sample) < sample_size:
                remaining_sample_size = sample_size - len(ecc_sample)
                random_companies = np.random.choice(unique_companies, size=remaining_sample_size, replace=False)
                
                for permco in random_companies:
                    if len(ecc_sample) >= sample_size:
                        break

                    company_rows = index_file[index_file['permco'] == permco]
                    company_name = company_rows.iloc[0]['company_name_TR']

                    ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]
                    print(f"Found {len(ecc_files)} files for permco {permco}")

                    for ecc_file in ecc_files:
                        if len(ecc_sample) >= sample_size:
                            break

                        # Match filename against the pattern
                        match = pattern.match(ecc_file)
                        if not match:
                            print(f"File name {ecc_file} does not match expected pattern. Skipping file.")
                            continue
                        
                        # Extract SE ID from the filename
                        se_id = int(match.group(2))
                        ecc_key = f"earnings_call_{permco}_{se_id}"
                        specific_row = company_rows[company_rows['SE_ID'] == se_id]
                        date = specific_row.iloc[0]['date'] if not specific_row.empty else 'Unknown'

                        with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                            text_content = file.read()

                        # Check if the word count is above the threshold (1600 words)
                        word_count = len(text_content.split())
                        if word_count < 1600:
                            excluded_count += 1
                            continue  # Skip this file if below the word count threshold

                        # Add to sample if word count is sufficient
                        if permco not in ecc_sample:
                            ecc_sample[permco] = {}
                        ecc_sample[permco][ecc_key] = {
                            'company_name': company_name,
                            'date': date,
                            'text_content': text_content
                        }
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        # Print the number of excluded calls
        print(f"Excluded {excluded_count} calls due to having fewer than 1600 words (below 0.5% percentile).")
        
        return ecc_sample


    def get_word_count_percentile(self, ecc_sample, percentile=1):
        """
        Calculates the word count at a specified percentile for the ECC sample.

        Parameters
        ----------
        ecc_sample : dict
            Dictionary containing the earnings call transcripts.
        percentile : float
            The percentile to calculate (default is 1 for the 1st percentile).

        Returns
        -------
        float
            The word count at the specified percentile.
        """
        # List to store word counts
        word_counts = []

        # Iterate through the sample and count words
        for permco, calls in ecc_sample.items():
            for ecc_key, ecc_data in calls.items():
                # Count words in the 'text_content' field
                word_count = len(ecc_data['text_content'].split())
                word_counts.append(word_count)

        # Calculate the percentile
        word_count_percentile = np.percentile(word_counts, percentile)
        print(f"The {percentile}th percentile word count is: {word_count_percentile}")
        return word_count_percentile
    
    def get_character_count_percentile(self, ecc_sample, percentile=1):
        """
        Calculates the character count at a specified percentile for the ECC sample.

        Parameters
        ----------
        ecc_sample : dict
            Dictionary containing the earnings call transcripts.
        percentile : float
            The percentile to calculate (default is 1 for the 1st percentile).

        Returns
        -------
        float
            The character count at the specified percentile.
        """
        # List to store character counts
        character_counts = []

        # Iterate through the sample and count characters
        for permco, calls in ecc_sample.items():
            for ecc_key, ecc_data in calls.items():
                # Count characters in the 'text_content' field
                character_count = len(ecc_data['text_content'])
                character_counts.append(character_count)

        # Calculate the percentile
        character_count_percentile = np.percentile(character_counts, percentile)
        print(f"The {percentile}th percentile character count is: {character_count_percentile}")
        return character_count_percentile