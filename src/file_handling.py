import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm  # Optional: For progress bars

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
            raise ValueError("File paths 'index_file_path' and 'folderpath_ecc' must be set in the config.")

    def read_index_file(self):
        """
        Reads the index file into a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The index file as a pandas DataFrame
        """
        print("Reading index file...")
        index_file = pd.read_csv(self.index_file_path, sep=";")
        return index_file

    def create_ecc_sample(self, sample_size):
        """
        Creates a sample of earnings call transcripts based on the sampling mode specified in the config,
        excluding calls with fewer than 1600 words.

        Parameters
        ----------
        sample_size : int
            The number of companies to include in the sample

        Returns
        -------
        dict
            A nested dictionary containing the sampled earnings call transcripts
        """
        # Read the index file
        index_file = self.read_index_file()

        # Initialize the ECC sample as a nested dictionary
        ecc_sample = {}
        excluded_count = 0  # Counter for excluded calls

        # Get all the files in the ECC folder
        all_files = os.listdir(self.folderpath_ecc)
        print(f"Total files found in ECC folder: {len(all_files)}")
        print(f"First 10 files in directory: {all_files[:10]}")

        # Regular expression pattern to match the expected filename format
        pattern = re.compile(r'^earnings_call_(\d+)_(\d+)\.txt$')

        # Get the sampling mode from the config, default to 'random_company'
        sampling_mode = self.config.get('sampling_mode', 'random_company')

        if sampling_mode == 'full_random':
            print("Sampling mode: full_random")
            # Implement 'full_random' sampling logic
            eligible_files = []
            for ecc_file in all_files:
                match = pattern.match(ecc_file)
                if not match:
                    print(f"File name {ecc_file} does not match expected pattern. Skipping file.")
                    continue
                permco, se_id = int(match.group(1)), int(match.group(2))
                specific_row = index_file[(index_file['permco'] == permco) & (index_file['SE_ID'] == se_id)]
                if specific_row.empty:
                    print(f"No index entry for permco {permco} and SE_ID {se_id}. Skipping file.")
                    continue
                date = specific_row.iloc[0]['date']
                company_name = specific_row.iloc[0]['company_name_TR']
                try:
                    with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                        text_content = file.read()
                except Exception as e:
                    print(f"Error reading file {ecc_file}: {e}")
                    continue
                word_count = len(text_content.split())
                if word_count < 1600:
                    excluded_count += 1
                    continue
                eligible_files.append((permco, se_id, company_name, date, text_content))

            if len(eligible_files) < sample_size:
                print(f"Only {len(eligible_files)} eligible transcripts available, which is less than the requested sample size of {sample_size}.")
                sample_size = len(eligible_files)

            sampled_files = np.random.choice(eligible_files, size=sample_size, replace=False)
            for permco, se_id, company_name, date, text_content in sampled_files:
                ecc_key = f"earnings_call_{permco}_{se_id}"
                if permco not in ecc_sample:
                    ecc_sample[permco] = {}
                ecc_sample[permco][ecc_key] = {
                    'permco': permco,
                    'se_id': se_id,
                    'company_name': company_name,
                    'date': date,
                    'text_content': text_content
                }

            print(f"Sampling completed. Total transcripts included: {len(sampled_files)}")
            print(f"Excluded {excluded_count} calls due to having fewer than 1600 words.")

        elif sampling_mode == 'random_company':
            print("Sampling mode: random_company")

            unique_companies = index_file['permco'].unique()
            total_unique_companies = len(unique_companies)
            print(f"Total unique companies available: {total_unique_companies}")

            # Check if there are enough unique companies
            if sample_size > total_unique_companies:
                raise ValueError(f"Requested sample size ({sample_size}) exceeds the number of unique companies available ({total_unique_companies}).")

            # Randomly select the desired number of unique companies without replacement
            sampled_companies = np.random.choice(unique_companies, size=sample_size, replace=False)
            print(f"Selected {len(sampled_companies)} unique companies for sampling.")

            # Iterate over each sampled company with a progress bar
            for permco in tqdm(sampled_companies, desc="Processing Companies"):
                company_rows = index_file[index_file['permco'] == permco]
                if company_rows.empty:
                    print(f"No entries found in index file for permco {permco}. Skipping company.")
                    continue

                company_name = company_rows.iloc[0]['company_name_TR']
                print(f"Processing company {permco}: {company_name}")

                # Find all ECC files for the current company
                ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]
                print(f"Found {len(ecc_files)} files for permco {permco}")

                # Iterate over each file for the company
                for ecc_file in ecc_files:
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

                    # Open and read text content
                    try:
                        with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                            text_content = file.read()
                    except Exception as e:
                        print(f"Error reading file {ecc_file}: {e}")
                        continue

                    # Check if the word count is above the threshold (1600 words)
                    word_count = len(text_content.split())
                    if word_count < 1600:
                        excluded_count += 1
                        continue  # Skip this file if below the word count threshold

                    # Add transcript to the sample
                    if permco not in ecc_sample:
                        ecc_sample[permco] = {}
                    ecc_sample[permco][ecc_key] = {
                        'permco': permco,
                        'se_id': se_id,
                        'company_name': company_name,
                        'date': date,
                        'text_content': text_content
                    }

            print(f"Sampling completed. Total transcripts included: {sum(len(calls) for calls in ecc_sample.values())}")
            print(f"Excluded {excluded_count} calls due to having fewer than 1600 words.")

        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

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
