# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:29:41 2024

@author: nikla
"""

#finance data integration exploration

import pandas as pd

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"
file_path_crsp_daily = "D:/daten_masterarbeit/crsp_daily_sample.csv"
file_path_crsp_monthly = "D:/daten_masterarbeit/crsp_monthly_sample.csv"
merged_file_path =  "D:/daten_masterarbeit/merged_topics_crsp_sample.csv"

#%% daily data sample creation
import pandas as pd

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"
file_path_crsp_daily = "D:/daten_masterarbeit/CRSP_daily_master_thesis.csv"
output_daily_sample = "D:/daten_masterarbeit/crsp_daily_sample.csv"


# Read the topics CSV file
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")

# Ensure 'permco' column is of type string
df_topics['permco'] = df_topics['permco'].astype(str)

# Extract unique permcos from df_topics
permcos = set(df_topics['permco'].unique())

# Define the chunksize (adjust based on your system's memory capacity)
chunksize = 10 ** 6  # This reads 1 million rows at a time

# Process the CRSP daily data in chunks
print("Processing CRSP/Daily data in chunks...")

# Initialize a flag for the header
header_saved = False

# Open the output file in write mode
with pd.option_context('mode.chained_assignment', None):
    for chunk in pd.read_csv(file_path_crsp_daily, chunksize=chunksize):
        # Ensure 'permco' column is of type string
        chunk['permco'] = chunk['permco'].astype(str)

        # Filter the chunk to include only permcos in df_topics
        filtered_chunk = chunk[chunk['permco'].isin(permcos)]

        # Append the filtered chunk to the output file
        if not filtered_chunk.empty:
            if not header_saved:
                filtered_chunk.to_csv(output_daily_sample, index=False, mode='w')
                header_saved = True
            else:
                filtered_chunk.to_csv(output_daily_sample, index=False, mode='a', header=False)
        print(f"Processed a chunk with {len(chunk)} rows, filtered down to {len(filtered_chunk)} rows.")

print(f"Filtered CRSP daily data saved to {output_daily_sample}")



#%% monthly data sample creation
import pandas as pd

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"
file_path_crsp_monthly = "D:/daten_masterarbeit/CRSP_monthly_Compustat_quarterly_merged.csv"
output_monthly_sample = "D:/daten_masterarbeit/crsp_monthly_sample.csv"

# Read the topics CSV file
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")

# Ensure 'permco' column is of type string
df_topics['permco'] = df_topics['permco'].astype(str)

# Extract unique permcos from df_topics
permcos = set(df_topics['permco'].unique())

# Define the chunksize (adjust based on your system's memory capacity)
chunksize = 10 ** 6  # This reads 1 million rows at a time

# Process the CRSP monthly data in chunks
print("Processing CRSP/Monthly data in chunks...")

# Initialize a flag for the header
header_saved = False

# Open the output file in write mode
with pd.option_context('mode.chained_assignment', None):
    for chunk in pd.read_csv(file_path_crsp_monthly, chunksize=chunksize):
        # Ensure 'permco' column is of type string
        chunk['permco'] = chunk['permco'].astype(str)

        # Filter the chunk to include only permcos in df_topics
        filtered_chunk = chunk[chunk['permco'].isin(permcos)]

        # Append the filtered chunk to the output file
        if not filtered_chunk.empty:
            if not header_saved:
                filtered_chunk.to_csv(output_monthly_sample, index=False, mode='w')
                header_saved = True
            else:
                filtered_chunk.to_csv(output_monthly_sample, index=False, mode='a', header=False)
        print(f"Processed a chunk with {len(chunk)} rows, filtered down to {len(filtered_chunk)} rows.")

print(f"Filtered CRSP monthly data saved to {output_monthly_sample}")




#%% continue processing of data 

#alter code, schauen was hier noch gebraucht wird 
"""
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")
df_crsp_daily = pd.read_csv(file_path_crsp_daily)
print("CRSP/Daily file loaded successfully.")
df_crsp_monthly = pd.read_csv(file_path_crsp_monthly)
print("CRSP/Monthly file loaded successfully.")

#monthly
# Select relevant columns from CRSP/Compustat data monthly
print("Selecting relevant columns from CRSP/Compustat data monthly...")
#cols_to_keep = ['date', 'permco', 'prc', 'vol', 'ret', 'gvkey','datadate', 'epsfxq', 'shrout']
cols_to_keep_monthly = ['date', 'permco', 'gvkey','datadate', 'epsfxq']
df_crsp_monthly = df_crsp_daily[cols_to_keep_monthly]

#daily
#select relevant columns from CRSP/Compustat data daily
print("Selecting relevant columns from CRSP/Compustat data daily...")
cols_to_keep_daily = ['date', 'permco', 'gvkey','prc','vol','ret','shrout']
df_crsp_daily = df_crsp_daily[cols_to_keep_daily]

#topics
#select relevant columns from topics
print("Selecting relevant columns from topics...")
cols_to_keep_topics = ['permco', 'call_id', 'date','filtered_topics', 'filtered_texts','similarity_to_average']
df_topics = df_topics[cols_to_keep_topics]


# If there are duplicate column names after merging, such as multiple datadate columns, rename them or drop duplicates
df_crsp_daily = df_crsp_daily.rename(columns={'datadate': 'datadate_crsp'})

# Merge the DataFrames on permco
print("Merging the DataFrames on permco...")
merged_df = pd.merge(df_topics, df_crsp_daily, on='permco', how='inner')

# Inspect the merged DataFrame
print(merged_df.head())

# Save the merged DataFrame if needed
print("Saving the merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)

"""



# Read the CSV files (using the sample files created above)
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")
df_crsp_daily = pd.read_csv(file_path_crsp_daily)
print("CRSP/Daily file loaded successfully.")
df_crsp_monthly = pd.read_csv(file_path_crsp_monthly)
print("CRSP/Monthly file loaded successfully.")

# Ensure 'permco' columns are of the same data type
df_topics['permco'] = df_topics['permco'].astype(str)
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)
df_crsp_monthly['permco'] = df_crsp_monthly['permco'].astype(str)

# Convert date columns to datetime format
df_topics['date'] = pd.to_datetime(df_topics['date'])  # Replace 'date' with the actual date column in df_topics if different
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'])
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'])
df_crsp_monthly['date'] = pd.to_datetime(df_crsp_monthly['date'])  # If 'date' column exists

# Create 'quarter' columns in df_topics and df_crsp_monthly
df_topics['quarter'] = df_topics['date'].dt.to_period('Q')
df_crsp_monthly['quarter'] = df_crsp_monthly['datadate'].dt.to_period('Q')

# Select relevant columns from CRSP/Compustat data
print("Selecting relevant columns from CRSP/Compustat data...")
cols_to_keep_monthly = ['permco', 'gvkey', 'datadate', 'epsfxq', 'quarter']
df_crsp_monthly = df_crsp_monthly[cols_to_keep_monthly]

# Select relevant columns from CRSP daily data
cols_to_keep_daily = ['date', 'permco', 'gvkey', 'prc', 'vol', 'ret', 'shrout']
df_crsp_daily = df_crsp_daily[cols_to_keep_daily]

# Merge df_topics and df_crsp_daily on 'permco' and 'date' matching 'date'
print("Merging df_topics and df_crsp_daily on 'permco' and 'date' == 'date'...")
merged_daily = pd.merge(df_topics, df_crsp_daily, left_on=['permco', 'date'], right_on=['permco', 'date'], how='left')

# Now, merge the monthly data into the merged_daily DataFrame based on 'permco' and 'quarter'
print("Merging monthly data into the daily merged DataFrame based on 'permco' and 'quarter'...")
final_merged_df = pd.merge(merged_daily, df_crsp_monthly, on=['permco', 'quarter'], how='left', suffixes=('_daily', '_monthly'))

# Inspect the final merged DataFrame
print(final_merged_df.head())

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
final_merged_df.to_csv(merged_file_path, index=False)