# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:29:41 2024

@author: nikla
"""

#finance data integration exploration

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



#%% Monthly data sample creation
import pandas as pd

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"
file_path_crsp_monthly = "D:/daten_masterarbeit/CRSP_monthly_Compustat_quarterly_merged.csv"
output_monthly_sample = "D:/daten_masterarbeit/crsp_monthly_sample.csv"

# Read the topics CSV file
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")

# Ensure 'permco' column is of type string and strip whitespaces
df_topics['permco'] = df_topics['permco'].astype(str).str.strip()

# Extract unique permcos from df_topics
permcos = set(df_topics['permco'].unique())
print(f"Number of unique permcos in topics: {len(permcos)}")

# Define the chunksize (adjust based on your system's memory capacity)
chunksize = 10 ** 6  # This reads 1 million rows at a time

# Process the CRSP monthly data in chunks
print("Processing CRSP/Monthly data in chunks...")

# Initialize a flag for the header
header_saved = False

# Initialize a set to collect all permcos in the filtered data
all_filtered_permcos = set()

# Open the output file in write mode
with pd.option_context('mode.chained_assignment', None):
    for chunk_index, chunk in enumerate(pd.read_csv(file_path_crsp_monthly, chunksize=chunksize)):
        print(f"Processing chunk {chunk_index + 1}")

        # Ensure 'permco' column exists
        if 'permco' not in chunk.columns:
            print("Error: 'permco' column not found in the monthly data chunk.")
            print("Available columns:", chunk.columns.tolist())
            break

        # Ensure 'permco' column is of type string and strip whitespaces
        chunk['permco'] = chunk['permco'].astype(str).str.strip()

        # Check for null permco values
        null_permco_rows = chunk[chunk['permco'].isnull()]
        if not null_permco_rows.empty:
            print(f"Found {len(null_permco_rows)} rows with null permco in chunk {chunk_index + 1}")
            # Optionally drop these rows
            chunk = chunk[chunk['permco'].notnull()]

        # Filter the chunk to include only permcos in df_topics
        filtered_chunk = chunk[chunk['permco'].isin(permcos)]

        # Collect permcos from the filtered chunk
        all_filtered_permcos.update(filtered_chunk['permco'].unique())

        # Append the filtered chunk to the output file
        if not filtered_chunk.empty:
            if not header_saved:
                filtered_chunk.to_csv(output_monthly_sample, index=False, mode='w')
                header_saved = True
            else:
                filtered_chunk.to_csv(output_monthly_sample, index=False, mode='a', header=False)
            print(f"Filtered chunk size: {len(filtered_chunk)} rows.")
        else:
            print("No matching permcos found in this chunk.")

print(f"Filtered CRSP monthly data saved to {output_monthly_sample}")

# After processing all chunks, compare the permcos
unexpected_permcos = all_filtered_permcos - permcos
if unexpected_permcos:
    print(f"Unexpected permcos found in filtered data: {unexpected_permcos}")
else:
    print("All permcos in filtered data match the permcos from df_topics.")



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

#%%
import pandas as pd
import pytz

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_final_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"
file_path_crsp_daily = "D:/daten_masterarbeit/crsp_daily_sample.csv"
file_path_crsp_monthly = "D:/daten_masterarbeit/crsp_monthly_sample.csv"
merged_file_path =  "D:/daten_masterarbeit/merged_topics_crsp_sample.csv"

# Read the CSV files
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

# Convert date columns to datetime format with explicit format and timezones
print("Converting date columns to datetime format...")

# For df_topics['date']
df_topics['date'] = pd.to_datetime(df_topics['date'], format='%d.%m.%Y %H:%M', errors='coerce')
df_topics = df_topics.dropna(subset=['date'])

# Adjust to New York time
df_topics['date'] = df_topics['date'].dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('America/New_York')
df_topics = df_topics.dropna(subset=['date'])  # Drop rows where timezone conversion failed

# Remove timezone info to make 'date' naive and normalize to remove time component
df_topics['date'] = df_topics['date'].dt.tz_localize(None).dt.normalize()

# For df_crsp_daily['date']
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'], errors='coerce')
df_crsp_daily = df_crsp_daily.dropna(subset=['date'])
df_crsp_daily['date'] = df_crsp_daily['date'].dt.normalize()

# Select relevant columns from topics
print("Selecting relevant columns from topics...")
cols_to_keep_topics = ['permco', 'call_id', 'date', 'filtered_topics', 'filtered_texts', 'similarity_to_average']
df_topics = df_topics[cols_to_keep_topics]

# Create 'quarter' columns in df_topics and df_crsp_monthly
df_topics['quarter'] = df_topics['date'].dt.to_period('Q')
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly.dropna(subset=['datadate'])
df_crsp_monthly['quarter'] = df_crsp_monthly['datadate'].dt.to_period('Q')

# Select relevant columns from CRSP/Compustat monthly data
print("Selecting relevant columns from CRSP/Compustat monthly data...")
cols_to_keep_monthly = ['permco', 'gvkey', 'datadate', 'epsfxq', 'quarter']
df_crsp_monthly = df_crsp_monthly[cols_to_keep_monthly]

# Select relevant columns from CRSP daily data
print("Selecting relevant columns from CRSP daily data...")
cols_to_keep_daily = ['date', 'permco', 'gvkey', 'prc', 'vol', 'ret', 'shrout']
df_crsp_daily = df_crsp_daily[cols_to_keep_daily]

# Merge df_topics and df_crsp_daily on 'permco' and 'date'
print("Merging df_topics and df_crsp_daily on 'permco' and 'date'...")
merged_daily = pd.merge(df_topics, df_crsp_daily, on=['permco', 'date'], how='left')

# Check if merge resulted in any matches
if merged_daily.empty:
    print("Merged DataFrame is empty after merging df_topics and df_crsp_daily.")
else:
    print("Number of rows in merged_daily:", len(merged_daily))

# Merge the monthly data into the merged_daily DataFrame
print("Merging monthly data into the daily merged DataFrame...")
final_merged_df = pd.merge(
    merged_daily,
    df_crsp_monthly,
    on=['permco', 'quarter'],
    how='left',
    suffixes=('_daily', '_monthly')
)

# Verify the final merged DataFrame
if final_merged_df.empty:
    print("Final merged DataFrame is empty after merging with monthly data.")
else:
    print("Number of rows in final_merged_df:", len(final_merged_df))
    print(final_merged_df.head())

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
final_merged_df.to_csv(merged_file_path, index=False)
