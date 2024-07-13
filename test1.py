import os
import pandas as pd

# Function to read the index file
def read_index_file(index_file_path):
    try:
        df = pd.read_csv(index_file_path, sep=';')
        return df
    except FileNotFoundError:
        print(f"Error: File '{index_file_path}' not found.")
        return None

# Function to create sample of earnings conference calls
def create_ecc_sample(sample_size, index_file, folderpath_ecc):
    ecc_sample = {}

    # Ensure sample size does not exceed number of unique companies (permco)
    unique_companies = index_file['permco'].unique()
    random_companies = np.random.choice(unique_companies, size=sample_size, replace=False)
    
    for permco in random_companies:
        # Get company details from index file
        company_info = index_file[index_file['permco'] == permco].iloc[0]
        company_name = company_info['company_name_TR']
        date = company_info['date']

        # Find all ECC files for this permco
        ecc_files = [f for f in os.listdir(folderpath_ecc) if f.startswith(f'earnings_call_{permco}_')]

        for ecc_file in ecc_files:
            se_id = ecc_file.split('_')[2]  # Extract SE_ID from filename
            ecc_key = f"earnings_call_{permco}_{se_id}"

            # Read the text content of the ECC file
            with open(os.path.join(folderpath_ecc, ecc_file), 'r') as file:
                text_content = file.read()

            # Construct the dictionary entry
            ecc_sample[ecc_key] = ((company_name, date), text_content)

    return ecc_sample

# Main function to run the script
def main():
    # Example paths and sample size
    index_file_path = '/mnt/data/list_earnings_call_transcripts.csv'
    folderpath_ecc = '/mnt/data/earnings_calls/'
    sample_size = 5  # Adjust sample size as needed, only needed at first run
    
    # Read the index file
    index_file = read_index_file(index_file_path)
    
    if index_file is not None:
        print("Index file loaded successfully.")
        
        # Create sample of earnings conference calls
        ecc_sample = create_ecc_sample(sample_size, index_file, folderpath_ecc)
        
        # Display the sample (for demonstration)
        print("\nHere is the sample of earnings conference calls:")
        for key, value in ecc_sample.items():
            print(f"Key: {key}")
            print(f"Company Info: {value[0]}")
            print(f"Text Content: {value[1][:100]}...")  # Displaying first 100 characters of text
            print()
    else:
        print("Failed to load index file.")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
