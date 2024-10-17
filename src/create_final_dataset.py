# create_final_dataset.py
import json
import pandas as pd
import numpy as np
from utils import process_topics, compute_similarity_to_average

# If git structure is not working properly:
fallback_config_path = "C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json"

# Load configuration variables from config.json
try: 
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        print("Config File Loaded.")
except: 
    with open(fallback_config_path, 'r') as config_file:
        config = json.load(config_file)
        print("Config File Loaded.")

topic_input_path = config['topics_input_path']
topic_output_path = config['topics_output_path']
topics_to_keep = config['topics_to_keep']
file_path_crsp_daily = config['file_path_crsp_daily']
file_path_crsp_monthly = config['file_path_crsp_monthly']
merged_file_path = config['merged_file_path']

# Process the topics and compute similarity to the average
print("Processing topics and computing similarity to the average call...")
processed_df = process_topics(topic_input_path, topic_output_path, topics_to_keep)
print(f"Processed DataFrame columns: {processed_df.columns}")

num_topics = processed_df['filtered_topics'].apply(lambda x: max(x) if x else 0).max() + 1
similarity_df = compute_similarity_to_average(processed_df, num_topics)

# Merge similarity back into the processed DataFrame
processed_df = processed_df.merge(similarity_df, on='call_id', how='left')

# Ensure 'permco' is a string in all DataFrames
processed_df['permco'] = processed_df['permco'].astype(str)

# Extract unique permcos from processed_df
permcos = set(processed_df['permco'].unique())

# Process the CRSP daily data
print("Processing CRSP/Daily data...")
chunksize = 10 ** 6
daily_data = []

for chunk in pd.read_csv(file_path_crsp_daily, chunksize=chunksize):
    chunk['permco'] = chunk['permco'].astype(str)
    filtered_chunk = chunk[chunk['permco'].isin(permcos)]
    if not filtered_chunk.empty:
        daily_data.append(filtered_chunk)

df_crsp_daily = pd.concat(daily_data, ignore_index=True)

# Process the CRSP monthly data
print("Processing CRSP/Monthly data...")
monthly_data = []
for chunk in pd.read_csv(file_path_crsp_monthly, chunksize=chunksize):
    chunk['permco'] = chunk['permco'].astype(str).str.strip()
    filtered_chunk = chunk[chunk['permco'].isin(permcos)]
    if not filtered_chunk.empty:
        monthly_data.append(filtered_chunk)

df_crsp_monthly = pd.concat(monthly_data, ignore_index=True)

# Ensure 'datadate' in df_crsp_monthly is in datetime format and remove NaNs
df_crsp_monthly = df_crsp_monthly[["datadate", "epsfxq", "permco", "siccd"]]
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
df_crsp_monthly['permco'] = df_crsp_monthly['permco'].astype(str)
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['permco'].notna()]

# Create 'epsfxq_next' by shifting 'epsfxq' within each 'permco'
df_crsp_monthly = df_crsp_monthly.sort_values(by=['permco', 'datadate']).reset_index(drop=True)
df_crsp_monthly['epsfxq_next'] = df_crsp_monthly.groupby('permco')['epsfxq'].shift(-1)

# Convert 'date' in processed_df to 'call_date' in datetime
processed_df['call_date'] = pd.to_datetime(processed_df['date'], utc=True, errors='coerce')
processed_df = processed_df.drop(columns=['date'])

# Remove rows with missing 'call_date' or 'permco' in processed_df
processed_df = processed_df[processed_df['call_date'].notna()]
processed_df = processed_df[processed_df['permco'].notna()]

# Convert 'call_date' to New York time and remove timezone information
processed_df['call_date'] = processed_df['call_date'].dt.tz_convert('America/New_York').dt.tz_localize(None)

# Remove timezone information from 'call_date' to match 'datadate'
processed_df['call_date'] = processed_df['call_date'].dt.tz_localize(None)

# Ensure 'permco' columns are of the same data type in both DataFrames
processed_df['permco'] = processed_df['permco'].astype(str)
df_crsp_monthly['permco'] = df_crsp_monthly['permco'].astype(str)

# **Updated Sorting: Sort by 'call_date' and 'permco'**
print("Ensuring both DataFrames are properly sorted...")
processed_df = processed_df.sort_values(by=['call_date', 'permco'], ascending=[True, True]).reset_index(drop=True)
df_crsp_monthly = df_crsp_monthly.sort_values(by=['datadate', 'permco'], ascending=[True, True]).reset_index(drop=True)

# Verify global sorting
print("Checking if 'call_date' is globally sorted in processed_df...")
is_call_date_sorted = processed_df['call_date'].is_monotonic_increasing
print(f"Is 'call_date' globally sorted? {is_call_date_sorted}")

print("Checking if 'datadate' is globally sorted in df_crsp_monthly...")
is_datadate_sorted = df_crsp_monthly['datadate'].is_monotonic_increasing
print(f"Is 'datadate' globally sorted? {is_datadate_sorted}")

# Proceed with the merge_asof operation
print("Merging topics with CRSP/Monthly data using merge_asof (backward)...")
merged_df = pd.merge_asof(
    processed_df,
    df_crsp_monthly[['permco', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd']],
    left_on='call_date',
    right_on='datadate',
    by='permco',
    direction='backward',
    allow_exact_matches=True
)
# Optionally, rename 'datadate' to 'fiscal_period_end' for clarity
merged_df.rename(columns={'datadate': 'fiscal_period_end'}, inplace=True)

# Ensure 'date' in CRSP daily is in datetime format
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'], errors='coerce')
df_crsp_daily = df_crsp_daily[df_crsp_daily['date'].notna()]
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)

# Merge with CRSP daily data on 'permco' and 'call_date' == 'date'
print("Merging with CRSP/Daily data...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'prc', 'shrout', 'ret', 'vol']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date'])

# Calculate future returns for CRSP daily data
print("Calculating future returns for CRSP/Daily data...")
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
        if i + 1 < n:
            ret_next_day[i] = ret_values[i+1]
        if i + 5 < n:
            ret_5_days[i] = np.prod(1 + ret_values[i+1:i+6]) - 1
        if i + 20 < n:
            ret_20_days[i] = np.prod(1 + ret_values[i+1:i+21]) - 1
        if i + 60 < n:
            ret_60_days[i] = np.prod(1 + ret_values[i+1:i+61]) - 1
    group['ret_next_day'] = ret_next_day
    group['ret_5_days'] = ret_5_days
    group['ret_20_days'] = ret_20_days
    group['ret_60_days'] = ret_60_days
    return group

df_crsp_daily = df_crsp_daily.groupby('permco').apply(compute_future_returns).reset_index(drop=True)

# Merge returns into the merged DataFrame
print("Merging returns into the merged DataFrame...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date', 'topics', 'text', 'consistent'])

# Rearrange columns to include 'epsfxq_next'
merged_df = merged_df[['permco', 'siccd', 'call_date', 'fiscal_period_end', 'filtered_topics', 'filtered_texts',
                       'prc', 'shrout', 'vol', 'ret', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days',
                       'epsfxq', 'epsfxq_next']]

# Fill NaNs in 'siccd' based on 'permco'
print("Filling NaNs in 'siccd' column...")
merged_df['siccd'] = merged_df.groupby('permco')['siccd'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Sort the final DataFrame by 'permco' and 'call_date' in ascending order
print("Sorting the final DataFrame by 'permco' and 'call_date'...")
merged_df = merged_df.sort_values(by=['permco', 'call_date'], ascending=[True, True]).reset_index(drop=True)

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
print(f"Final merged DataFrame saved to {merged_file_path}")
