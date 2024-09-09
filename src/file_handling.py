import os
import pandas as pd
import numpy as np

class FileHandler:
    """
    Class to handle file operations

    Attributes
    ----------
    index_file_path : str
        Path to the index file
    folderpath_ecc : str
        Path to the folder containing the earnings call transcripts
    """

    def __init__(self, index_file_path=None, folderpath_ecc=None):
        """
        Initializes the FileHandler

        Parameters
        ----------
        index_file_path : str, optional
            Path to the index file
        folderpath_ecc : str, optional
            Path to the folder containing the earnings call transcripts
        """
        # Set the file paths
        self.index_file_path = index_file_path
        self.folderpath_ecc = folderpath_ecc

    def read_index_file(self):
        """
        Reads the index file into a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The index file as a pandas DataFrame
        """
        # Check if the index file path is set
        if self.index_file_path is None:
            raise ValueError("Index file path is not set.")
        # Read the index file
        index_file = pd.read_csv(self.index_file_path, sep=";")
        return index_file

    def set_file_paths(self, index_file_path, folderpath_ecc):
        """
        Sets the file paths

        Parameters
        ----------
        index_file_path : str
            Path to the index file
        folderpath_ecc : str
            Path to the folder containing the earnings call transcripts
        """
        # Set the file paths
        self.index_file_path = index_file_path
        self.folderpath_ecc = folderpath_ecc

    def create_ecc_sample(self, sample_size):
        """
        Creates a sample of earnings call transcripts

        Parameters
        ----------
        sample_size : int
            The number of unique companies to include in the sample

        Returns
        -------
        dict
            A dictionary where the keys are the permco and the values are dictionaries with the following keys:
                - company_name : str
                - date : str
                - text_content : str
        """
        # Check if the file paths are set
        if self.index_file_path is None or self.folderpath_ecc is None:
            raise ValueError("File paths must be set before creating ECC sample.")
        
        # Read the index file
        index_file = self.read_index_file()
        # Initialize the ECC sample
        ecc_sample = {}

        # Get the unique companies
        unique_companies = index_file['permco'].unique()
        # Limit the sample size to the number of unique companies
        sample_size = min(sample_size, len(unique_companies))
        # Select a random subset of the unique companies
        random_companies = np.random.choice(unique_companies, size=sample_size, replace=False)

        # Get all the files in the ECC folder
        all_files = os.listdir(self.folderpath_ecc)
        print("First 10 files in directory:", all_files[:10])

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
                ecc_sample[permco][ecc_key] = (company_name, date, text_content)

        return ecc_sample

