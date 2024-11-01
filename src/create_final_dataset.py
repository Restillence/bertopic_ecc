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
except FileNotFoundError: 
    with open(fallback_config_path, 'r') as config_file:
        config = json.load(config_file)
        print("Config File Loaded.")

topic_input_path = config['topics_input_path']
topic_output_path = config['topics_output_path']
topics_to_keep = config['topics_to_keep']
file_path_crsp_daily = config['file_path_crsp_daily']
file_path_crsp_monthly = config['file_path_crsp_monthly']
merged_file_path = config['merged_file_path']
topic_threshold_percentage = config['topic_threshold_percentage']  # Add this to your config

#%% Process the topics
print("Processing topics...")
processed_df = process_topics(topic_input_path, topic_output_path, topics_to_keep, topic_threshold_percentage)
print(f"Processed DataFrame columns: {processed_df.columns}")

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

# Ensure 'date' in CRSP daily is in datetime format
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'], errors='coerce')
df_crsp_daily = df_crsp_daily[df_crsp_daily['date'].notna()]

# Ensure 'permco' and 'gvkey' are strings
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)
df_crsp_daily['gvkey'] = df_crsp_daily['gvkey'].astype(str)

# Handle missing 'gvkey' values in df_crsp_daily
missing_gvkey_daily = df_crsp_daily['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in df_crsp_daily before filling: {missing_gvkey_daily}")

if missing_gvkey_daily > 0:
    print("Filling missing 'gvkey' values in df_crsp_daily based on the most frequent 'gvkey' per 'permco'...")
    # Compute the most frequent gvkey per permco
    most_common_gvkey_per_permco_daily = df_crsp_daily.groupby('permco')['gvkey'].apply(
        lambda x: x.value_counts().idxmax() if x.notna().any() else np.nan
    )
    # Map the most frequent gvkey back to df_crsp_daily
    df_crsp_daily['gvkey'] = df_crsp_daily.apply(
        lambda row: most_common_gvkey_per_permco_daily[row['permco']] if pd.isna(row['gvkey']) else row['gvkey'],
        axis=1
    )
    # Check missing 'gvkey's again
    missing_gvkey_daily = df_crsp_daily['gvkey'].isna().sum()
    print(f"Number of missing 'gvkey' values in df_crsp_daily after filling: {missing_gvkey_daily}")

# Convert 'date' in processed_df to 'call_date' in datetime
processed_df['call_date'] = pd.to_datetime(processed_df['date'], utc=True, errors='coerce')
processed_df = processed_df.drop(columns=['date'])

# Remove rows with missing 'call_date' or 'permco'
processed_df = processed_df[processed_df['call_date'].notna()]
processed_df = processed_df[processed_df['permco'].notna()]

# Convert 'call_date' to New York time and remove timezone information
processed_df['call_date'] = processed_df['call_date'].dt.tz_convert('America/New_York').dt.tz_localize(None)

# Normalize 'call_date' and 'date' to remove time component for merging
processed_df['call_date'] = processed_df['call_date'].dt.normalize()
df_crsp_daily['date'] = df_crsp_daily['date'].dt.normalize()

#%% Merge processed_df with df_crsp_daily to get 'gvkey' into processed_df
print("Merging processed_df on permco and call_date with df_crsp_daily to get 'gvkey'...")
processed_df = pd.merge(
    processed_df,
    df_crsp_daily[['permco', 'date', 'gvkey']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
processed_df = processed_df.drop(columns=['date'])

# Handle missing 'gvkey' values in processed_df
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in processed_df after merging: {missing_gvkey}")

# Fill missing 'gvkey' values in processed_df based on the most frequent 'gvkey' per 'permco'
print("Filling missing 'gvkey' values in processed_df based on the most frequent 'gvkey' per 'permco'...")
# Compute the most frequent gvkey per permco from df_crsp_daily (since it has more data)
most_common_gvkey_per_permco = df_crsp_daily.groupby('permco')['gvkey'].apply(
    lambda x: x.value_counts().idxmax() if x.notna().any() else np.nan
)

# Map the most frequent gvkey back to processed_df
processed_df['gvkey'] = processed_df.apply(
    lambda row: most_common_gvkey_per_permco[row['permco']] if pd.isna(row['gvkey']) else row['gvkey'],
    axis=1
)

# Now, check how many NaNs are left
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in processed_df after filling: {missing_gvkey}")

# Remove rows with missing 'gvkey' if necessary
processed_df = processed_df[processed_df['gvkey'].notna()]
processed_df['gvkey'] = processed_df['gvkey'].astype(str)

# Process the CRSP monthly data
print("Processing CRSP/Monthly data...")
chunksize = 10 ** 6
monthly_data = []
for chunk in pd.read_csv(file_path_crsp_monthly, chunksize=chunksize):
    chunk['gvkey'] = chunk['gvkey'].astype(str)
    chunk['permco'] = chunk['permco'].astype(str)
    filtered_chunk = chunk[chunk['gvkey'].isin(processed_df['gvkey'].unique())]
    if not filtered_chunk.empty:
        monthly_data.append(chunk)

df_crsp_monthly = pd.concat(monthly_data, ignore_index=True)

# Ensure 'datadate' in df_crsp_monthly is in datetime format and remove NaNs
df_crsp_monthly = df_crsp_monthly[["datadate", "epsfxq", "gvkey", "siccd", "permco"]]
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
df_crsp_monthly['gvkey'] = df_crsp_monthly['gvkey'].astype(str)
df_crsp_monthly['permco'] = df_crsp_monthly['permco'].astype(str)

# Handle missing 'siccd' values in df_crsp_monthly
missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
print(f"Number of missing 'siccd' values in df_crsp_monthly before filling: {missing_siccd_monthly}")

if missing_siccd_monthly > 0:
    print("Filling missing 'siccd' values in df_crsp_monthly based on the most frequent 'siccd' per 'permco'...")
    # Compute the most frequent siccd per permco
    most_common_siccd_per_permco_monthly = df_crsp_monthly.groupby('permco')['siccd'].apply(
        lambda x: x.value_counts().idxmax() if x.notna().any() else np.nan
    )
    # Map the most frequent siccd back to df_crsp_monthly
    df_crsp_monthly['siccd'] = df_crsp_monthly.apply(
        lambda row: most_common_siccd_per_permco_monthly[row['permco']] if pd.isna(row['siccd']) else row['siccd'],
        axis=1
    )
    # Check missing 'siccd's again
    missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
    print(f"Number of missing 'siccd' values in df_crsp_monthly after filling: {missing_siccd_monthly}")

# Ensure 'call_date' and 'datadate' are in datetime format
processed_df['call_date'] = pd.to_datetime(processed_df['call_date'], errors='coerce')
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')

# Remove rows with missing dates
processed_df = processed_df[processed_df['call_date'].notna()]
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]

# Ensure both DataFrames are sorted properly
print("Ensuring both DataFrames are properly sorted...")
processed_df = processed_df.sort_values(by=['call_date', 'gvkey']).reset_index(drop=True)
df_crsp_monthly = df_crsp_monthly.sort_values(by=['datadate', 'gvkey']).reset_index(drop=True)

# Verify that 'call_date' is globally sorted
print("Checking if 'call_date' is globally sorted in processed_df...")
is_call_date_sorted = processed_df['call_date'].is_monotonic_increasing
print(f"Is 'call_date' globally sorted? {is_call_date_sorted}")

print("Checking if 'datadate' is globally sorted in df_crsp_monthly...")
is_datadate_sorted = df_crsp_monthly['datadate'].is_monotonic_increasing
print(f"Is 'datadate' globally sorted? {is_datadate_sorted}")

# Proceed with the merge if sorting checks pass
if is_call_date_sorted and is_datadate_sorted:
    # Merge using merge_asof with direction='backward' to get 'epsfxq' and 'siccd' using 'gvkey'
    print("Merging processed_df with df_crsp_monthly using 'gvkey' and merge_asof...")
    print(f"Number of rows in processed_df before merging: {len(processed_df)}")
    
    # Exclude 'permco' from df_crsp_monthly to avoid column conflict
    merged_df = pd.merge_asof(
        processed_df,
        df_crsp_monthly[['gvkey', 'datadate', 'epsfxq', 'siccd']],  # Exclude 'permco' here
        left_on='call_date',
        right_on='datadate',
        by='gvkey',
        direction='backward',
        allow_exact_matches=True
    )
    
    print(f"Number of rows in merged_df after merging: {len(merged_df)}")
else:
    print("Cannot proceed with merge_asof because DataFrames are not properly sorted.")
    exit()

# Continue with the rest of your code
num_nan_siccd = merged_df['siccd'].isna().sum()
print(f"Number of rows with NaN 'siccd': {num_nan_siccd}")

# Fill missing 'siccd' values based on 'permco' using df_crsp_monthly
print("Filling missing 'siccd' values based on the most frequent 'siccd' per 'permco' from df_crsp_monthly...")
siccd_mapping = df_crsp_monthly.groupby('permco')['siccd'].apply(
    lambda x: x.value_counts().idxmax() if x.notna().any() else np.nan
)

# Ensure 'permco' is present in 'merged_df'
if 'permco' not in merged_df.columns:
    print("Error: 'permco' column is missing in 'merged_df'.")
else:
    merged_df['siccd'] = merged_df.apply(
        lambda row: siccd_mapping[row['permco']] if pd.isna(row['siccd']) and row['permco'] in siccd_mapping else row['siccd'],
        axis=1
    )

# Now, check how many NaNs are left
num_nan_siccd = merged_df['siccd'].isna().sum()
print(f"Number of rows with NaN 'siccd' after filling from df_crsp_monthly: {num_nan_siccd}")

# Proceed without dropping rows
if num_nan_siccd > 0:
    print(f"Proceeding with {num_nan_siccd} rows with NaN 'siccd'. Similarity to industry average will be NaN for these rows.")

# Optionally, rename 'datadate' to 'fiscal_period_end' for clarity
merged_df.rename(columns={'datadate': 'fiscal_period_end'}, inplace=True)

# Now that 'siccd' is available, compute similarities including similarity to industry average
print("Computing similarities to overall and industry averages...")
num_topics = merged_df['filtered_topics'].apply(lambda x: max(x) if x else 0).max() + 1

# Ensure 'filtered_topics' is evaluated as lists
merged_df['filtered_topics'] = merged_df['filtered_topics'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Convert 'siccd' to integer (if possible)
merged_df['siccd'] = merged_df['siccd'].astype(float).astype('Int64')

# Compute similarities
similarity_df = compute_similarity_to_average(merged_df, num_topics)

# Merge similarities back into merged_df
merged_df = merged_df.merge(similarity_df, on='call_id', how='left')

# Proceed with merging with CRSP daily data and calculating future returns
# Ensure 'call_date' and 'date' are date-only (no time component)
merged_df['call_date'] = pd.to_datetime(merged_df['call_date']).dt.normalize()
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date']).dt.normalize()

# Ensure 'permco' columns are of the same data type
merged_df['permco'] = merged_df['permco'].astype(str)
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)

# Merge with CRSP daily data on 'permco' and 'call_date' == 'date'
print("Merging with CRSP/Daily data after normalizing dates...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'prc', 'shrout', 'ret', 'vol']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date'])

# Convert 'ret' to numeric, handling non-numeric values
def clean_ret(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

df_crsp_daily['ret'] = df_crsp_daily['ret'].apply(clean_ret)
print(f"\nNumber of NaNs in 'ret' after cleaning: {df_crsp_daily['ret'].isna().sum()}")

# Proceed with computing future returns
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
        if i + 1 < n and not np.isnan(ret_values[i+1]):
            ret_next_day[i] = ret_values[i+1]
        if i + 5 < n and not np.isnan(ret_values[i+1:i+6]).any():
            ret_5_days[i] = np.prod(1 + ret_values[i+1:i+6]) - 1
        if i + 20 < n and not np.isnan(ret_values[i+1:i+21]).any():
            ret_20_days[i] = np.prod(1 + ret_values[i+1:i+21]) - 1
        if i + 60 < n and not np.isnan(ret_values[i+1:i+61]).any():
            ret_60_days[i] = np.prod(1 + ret_values[i+1:i+61]) - 1
    group['ret_next_day'] = ret_next_day
    group['ret_5_days'] = ret_5_days
    group['ret_20_days'] = ret_20_days
    group['ret_60_days'] = ret_60_days
    return group

df_crsp_daily = df_crsp_daily.groupby('permco').apply(compute_future_returns).reset_index(drop=True)

# Verify future returns
print("\nSummary statistics of future returns in df_crsp_daily:")
print(df_crsp_daily[['ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days']].describe())

# Merge future returns into merged_df
print("Merging future returns into the merged DataFrame...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date', 'topics', 'text', 'consistent'], errors='ignore')

# Rearrange columns to include 'epsfxq' and similarity measures
merged_df = merged_df[['gvkey', 'permco', 'siccd', 'call_id', 'call_date', 'fiscal_period_end', 'filtered_topics', 'filtered_texts',
                       'prc', 'shrout', 'vol', 'ret', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days',
                       'epsfxq', 'similarity_to_overall_average', 'similarity_to_industry_average',"similarity_to_company_average"]]

# Sort the final DataFrame by 'gvkey' and 'call_date' in ascending order
print("Sorting the final DataFrame by 'gvkey' and 'call_date'...")
merged_df = merged_df.sort_values(by=['gvkey', 'call_date']).reset_index(drop=True)

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
print(f"Final merged DataFrame saved to {merged_file_path}")

#calculate the means of the similarity measures
print("Calculating means of similarity measures...")
print("Mean of similarity_to_overall_average:", merged_df['similarity_to_overall_average'].mean())
print("Mean of similarity_to_industry_average:", merged_df['similarity_to_industry_average'].mean())
print("Mean of similarity_to_company_average:", merged_df['similarity_to_company_average'].mean())