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

#%% read files, create dfs (for the moment on small sample)
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


#%% remove nans in datadate and drop unneccssary columns

df_crsp_monthly = df_crsp_monthly[["datadate", "epsfxq", "permco"]]

# Remove rows from df_crsp_monthly where 'datadate' contains NaNs
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]


#%% convert to ny time

# Parse 'date' as datetime with UTC timezone
df_topics['date'] = pd.to_datetime(df_topics['date'], utc=True, errors='coerce')

# Convert to New York time (Eastern Time Zone)
df_topics['date'] = df_topics['date'].dt.tz_convert('America/New_York')

# Optionally, remove timezone information if not needed
df_topics['date'] = df_topics['date'].dt.tz_localize(None)

#%% merge df crsp monthly and topics df
import pandas as pd

# Ensure 'date' in df_topics is in datetime format (with time)
df_topics['date'] = pd.to_datetime(df_topics['date'], errors='coerce')

# Calculate the previous quarter end date for each 'date'
df_topics['quarter_end_date'] = df_topics['date'].apply(lambda x: (x - pd.offsets.QuarterEnd(n=1)).normalize())

# Ensure 'datadate' in df_crsp_monthly is in datetime format and date-only
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce').dt.normalize()

# Remove rows where 'datadate' contains NaNs
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]

# Merge the DataFrames on 'permco' and 'quarter_end_date' == 'datadate'
merged_df = pd.merge(
    df_topics,
    df_crsp_monthly,
    left_on=['permco', 'quarter_end_date'],
    right_on=['permco', 'datadate'],
    how='left'
)

# Drop 'datadate' column after merging if desired
merged_df = merged_df.drop(columns=['datadate'])

#%% crsp daily to datetime
import pandas as pd
import numpy as np

# Ensure 'date' in df_crsp_daily is in datetime format
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'], format='%d%b%Y', errors='coerce')

# Handle any parsing errors
df_crsp_daily = df_crsp_daily[df_crsp_daily['date'].notna()]


#%% merge topics and crsp daily
# Ensure 'permco' columns are of the same data type
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)
merged_df['permco'] = merged_df['permco'].astype(str)

#rename of date column to call date
merged_df["call_date"] = merged_df["date"]
merged_df = merged_df.drop(columns=['date'])

# Ensure 'call_date' in merged_df is date-only
merged_df['call_date'] = pd.to_datetime(merged_df['call_date']).dt.normalize()

# Merge the DataFrames on 'permco' and 'call_date' == 'date'
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'prc', 'shrout', 'ret', 'vol']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)

# Drop 'date' column from df_crsp_daily if desired
merged_df = merged_df.drop(columns=['date'])

#%% create return variables
df_crsp_daily['ret'] = pd.to_numeric(df_crsp_daily['ret'], errors='coerce')
df_crsp_daily = df_crsp_daily.sort_values(['permco', 'date']).reset_index(drop=True)

def compute_future_returns(group):
    group = group.sort_values('date').reset_index(drop=True)
    n = len(group)
    ret_next_day = np.full(n, np.nan)
    ret_5_days = np.full(n, np.nan)
    ret_20_days = np.full(n, np.nan)
    ret_60_days = np.full(n, np.nan)
    ret_values = group['ret'].values
    for i in range(n):
        # ret_next_day
        if i + 1 < n:
            ret_next_day[i] = ret_values[i+1]
        # ret_5_days
        if i + 5 < n:
            ret_5_days[i] = np.prod(1 + ret_values[i+1:i+6]) - 1
        # ret_20_days
        if i + 20 < n:
            ret_20_days[i] = np.prod(1 + ret_values[i+1:i+21]) - 1
        # ret_60_days
        if i + 60 < n:
            ret_60_days[i] = np.prod(1 + ret_values[i+1:i+61]) - 1
    group['ret_next_day'] = ret_next_day
    group['ret_5_days'] = ret_5_days
    group['ret_20_days'] = ret_20_days
    group['ret_60_days'] = ret_60_days
    return group

df_crsp_daily = df_crsp_daily.groupby('permco').apply(compute_future_returns).reset_index(drop=True)

# **Step 4: Merge the Future Returns into the Merged DataFrame**
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date'])


merged_file_path =  "D:/daten_masterarbeit/merged_topics_crsp_sample.csv"
# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
