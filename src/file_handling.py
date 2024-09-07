import os
import pandas as pd
import numpy as np

class FileHandler:
    def __init__(self, index_file_path=None, folderpath_ecc=None):
        self.index_file_path = index_file_path
        self.folderpath_ecc = folderpath_ecc

    def read_index_file(self):
        if self.index_file_path is None:
            raise ValueError("Index file path is not set.")
        index_file = pd.read_csv(self.index_file_path, sep=";")
        return index_file

    def set_file_paths(self, index_file_path, folderpath_ecc):
        self.index_file_path = index_file_path
        self.folderpath_ecc = folderpath_ecc

    def create_ecc_sample(self, sample_size):
        if self.index_file_path is None or self.folderpath_ecc is None:
            raise ValueError("File paths must be set before creating ECC sample.")
        
        index_file = self.read_index_file()
        ecc_sample = {}

        unique_companies = index_file['permco'].unique()
        sample_size = min(sample_size, len(unique_companies))
        random_companies = np.random.choice(unique_companies, size=sample_size, replace=False)

        all_files = os.listdir(self.folderpath_ecc)
        print("First 10 files in directory:", all_files[:10])

        print("Starting to create ECC sample...")
        for permco in random_companies:
            company_rows = index_file[index_file['permco'] == permco]
            company_name = company_rows.iloc[0]['company_name_TR']

            ecc_files = [f for f in all_files if f.startswith(f'earnings_call_{permco}_')]

            print(f"Found {len(ecc_files)} files for permco {permco}")

            for ecc_file in ecc_files:
                se_id = ecc_file.split('_')[3].replace('.txt', '')
                ecc_key = f"earnings_call_{permco}_{se_id}"

                specific_row = company_rows[company_rows['SE_ID'] == int(se_id)]
                if not specific_row.empty:
                    date = specific_row.iloc[0]['date']
                else:
                    date = 'Unknown'
                    print(f"SE_ID {se_id} not found in index file for permco {permco}")

                with open(os.path.join(self.folderpath_ecc, ecc_file), 'r', encoding='utf-8') as file:
                    text_content = file.read()

                if permco not in ecc_sample:
                    ecc_sample[permco] = {}
                ecc_sample[permco][ecc_key] = (company_name, date, text_content)

        return ecc_sample
