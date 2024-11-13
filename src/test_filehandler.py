# test_filehandler.py
import os
import json
from file_handling import FileHandler

# Load configuration
config = {
    "index_file_path": "D:/daten_masterarbeit/list_earnings_call_transcripts.csv",
    "folderpath_ecc": "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/",
    "sampling_mode": "full_random",
    "sample_size": 1000
    # Add other config parameters as necessary
}

# Initialize the FileHandler with the config
file_handler = FileHandler(config)

# Create a sample of earnings calls
ecc_sample = file_handler.create_ecc_sample(sample_size=config["sample_size"])

# Calculate and print the 1st percentile word count
percentile_word_count = file_handler.get_word_count_percentile(ecc_sample, percentile=0.5)
print(f"1st percentile word count: {percentile_word_count}")


# Calculate and print the 1st percentile character count
percentile_character_count = file_handler.get_character_count_percentile(ecc_sample, percentile=0.5)
print(f"1st percentile character count: {percentile_character_count}")