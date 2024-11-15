import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm  # Optional: For progress bars

class FileHandler:
    """
    Class to handle file operations related to sampling earnings call transcripts.
    
    Attributes
    ----------
    config : dict
        Configuration dictionary containing file paths and settings.
    """

    def __init__(self, config):
        """
        Initializes the FileHandler.

        Parameters
        ----------
        config : dict
            Configuration dictionary with the following keys:
            - 'index_file_path': Path to the index CSV file.
            - 'folderpath_ecc': Path to the folder containing ECC files.
            - 'sampling_mode': 'full_random' or 'random_company'.
        """
        # Set the configuration
        self.config = config

        # Extract file paths and settings from config
        self.index_file_path = self.config.get('index_file_path')
        self.folderpath_ecc = self.config.get('folderpath_ecc')
        self.sampling_mode = self.config.get('sampling_mode', 'random_company')

        # Validate essential configuration parameters
        if not self.index_file_path:
            raise ValueError("Configuration must include 'index_file_path'.")
        if not self.folderpath_ecc:
            raise ValueError("Configuration must include 'folderpath_ecc'.")
        if self.sampling_mode not in ['full_random', 'random_company']:
            raise ValueError("Sampling mode must be either 'full_random' or 'random_company'.")

    def read_index_file(self):
        """
        Reads the index file into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The index file as a pandas DataFrame.
        """
        print("Reading index file...")
        try:
            index_file = pd.read_csv(self.index_file_path, sep=";")
            print(f"Index file successfully read with {len(index_file)} entries.")
            return index_file
        except Exception as e:
            print(f"Error reading index file: {e}")
            raise

    def create_ecc_sample(self, sample_size):
        """
        Creates a sample of earnings call transcripts based on the sampling mode specified in the config,
        excluding calls with fewer than 1600 words.

        Parameters
        ----------
        sample_size : int
            - If `sampling_mode` is 'full_random', represents the number of transcripts to sample.
            - If `sampling_mode` is 'random_company', represents the number of unique companies to sample.

        Returns
        -------
        dict
            A nested dictionary containing the sampled earnings call transcripts.
            Structure:
            {
                permco1: {
                    'earnings_call_permco1_seid1': { ... },
                    'earnings_call_permco1_seid2': { ... },
                    ...
                },
                permco2: {
                    'earnings_call_permco2_seid3': { ... },
                    ...
                },
                ...
            }
        """
        print("Creating ECC sample...")

        # Read the index file
        index_file = self.read_index_file()

        # Initialize the ECC sample as a nested dictionary
        ecc_sample = {}
        excluded_count = 0  # Counter for excluded calls
        companies_with_no_transcripts = 0  # Counter for companies with no eligible transcripts

        # Get all the files in the ECC folder
        try:
            all_files = os.listdir(self.folderpath_ecc)
            print(f"Total files found in ECC folder: {len(all_files)}")
            if len(all_files) > 10:
                print(f"First 10 files in directory: {all_files[:10]}")
            else:
                print(f"Files in directory: {all_files}")
        except Exception as e:
            print(f"Error accessing ECC folder: {e}")
            raise

        # Regular expression pattern to match the expected filename format
        pattern = re.compile(r'^earnings_call_(\d+)_(\d+)\.txt$')

        if self.sampling_mode == 'full_random':
            print("Sampling mode: full_random")
            # Implement 'full_random' sampling logic
            eligible_files = []
            for ecc_file in all_files:
                match = pattern.match(ecc_file)
                if not match:
                    continue  # Skip files that don't match the pattern
                permco, se_id = int(match.group(1)), int(match.group(2))
                specific_row = index_file[(index_file['permco'] == permco) & (index_file['SE_ID'] == se_id)]
                if specific_row.empty:
                    continue  # Skip if no matching index entry
                date = specific_row.iloc[0]['date']
                company_name = specific_row.iloc[0]['company_name_TR']
                try:
                    with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                        text_content = file.read()
                except:
                    excluded_count += 1
                    continue  # Skip file if there's an error reading it
                word_count = len(text_content.split())
                if word_count < 1600:
                    excluded_count += 1
                    continue  # Skip if below word count threshold
                eligible_files.append((permco, se_id, company_name, date, text_content))

            if len(eligible_files) < sample_size:
                print(f"Only {len(eligible_files)} eligible transcripts available, which is less than the requested sample size of {sample_size}.")
                sample_size = len(eligible_files)

            # Randomly select the desired number of transcripts without replacement
            sampled_indices = np.random.choice(len(eligible_files), size=sample_size, replace=False)
            for idx in sampled_indices:
                permco, se_id, company_name, date, text_content = eligible_files[idx]
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

            print("\nSampling completed.")
            print(f"Total transcripts included: {len(sampled_indices)}")
            print(f"Excluded {excluded_count} calls due to having fewer than 1600 words.")

        elif self.sampling_mode == 'random_company':
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
                    companies_with_no_transcripts += 1
                    continue  # Skip company if no entries found

                company_name = company_rows.iloc[0]['company_name_TR']

                # Find all ECC files for the current company
                ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]

                # Iterate over each file for the company
                for ecc_file in ecc_files:
                    # Match filename against the pattern
                    match = pattern.match(ecc_file)
                    if not match:
                        continue  # Skip file if pattern does not match

                    # Extract SE ID from the filename
                    se_id = int(match.group(2))
                    ecc_key = f"earnings_call_{permco}_{se_id}"
                    specific_row = company_rows[company_rows['SE_ID'] == se_id]
                    date = specific_row.iloc[0]['date'] if not specific_row.empty else 'Unknown'

                    # Open and read text content
                    try:
                        with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                            text_content = file.read()
                    except:
                        excluded_count += 1
                        continue  # Skip file if there's an error reading it

                    # Check if the word count is above the threshold (1600 words)
                    word_count = len(text_content.split())
                    if word_count < 1600:
                        excluded_count += 1
                        continue  # Skip if below word count threshold

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

            # Summary of sampling
            total_transcripts = sum(len(calls) for calls in ecc_sample.values())
            print("\nSampling completed.")
            print(f"Total transcripts included: {total_transcripts}")
            print(f"Excluded {excluded_count} calls due to having fewer than 1600 words.")
            if companies_with_no_transcripts > 0:
                print(f"Skipped {companies_with_no_transcripts} companies due to no eligible transcripts.")

        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

        return ecc_sample

    def get_word_count_percentile(self, ecc_sample, percentile=1):
        """
        Calculates the word count at a specified percentile for the ECC sample.

        Parameters
        ----------
        ecc_sample : dict
            Dictionary containing the earnings call transcripts.
        percentile : float, optional
            The percentile to calculate (default is 1 for the 1st percentile).

        Returns
        -------
        float
            The word count at the specified percentile.
        """
        word_counts = [
            len(ecc_data['text_content'].split())
            for calls in ecc_sample.values()
            for ecc_data in calls.values()
        ]
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
        percentile : float, optional
            The percentile to calculate (default is 1 for the 1st percentile).

        Returns
        -------
        float
            The character count at the specified percentile.
        """
        character_counts = [
            len(ecc_data['text_content'])
            for calls in ecc_sample.values()
            for ecc_data in calls.values()
        ]
        character_count_percentile = np.percentile(character_counts, percentile)
        print(f"The {percentile}th percentile character count is: {character_count_percentile}")
        return character_count_percentile
