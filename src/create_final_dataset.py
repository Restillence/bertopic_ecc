# create_final_dataset.py

import json
import pandas as pd
import numpy as np
import sys
from utils import process_topics, compute_similarity_to_average

# Function to verify if a DataFrame is sorted within each group
def is_sorted_within_group(df, group_col, sort_col):
    return df.groupby(group_col)[sort_col].apply(lambda x: x.is_monotonic_increasing).all()

# Load configuration variables from config.json
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        print("Config File Loaded from 'config.json'.")
except FileNotFoundError:
    fallback_config_path = "C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json"
    with open(fallback_config_path, 'r') as config_file:
        config = json.load(config_file)
        print(f"Config File Loaded from fallback path: {fallback_config_path}")

# Extract configuration parameters
topic_input_path = config['topics_input_path']
topic_output_path = config['topics_output_path']
topics_to_keep = config['topics_to_keep']
file_path_crsp_daily = config['file_path_crsp_daily']
file_path_crsp_monthly = config['file_path_crsp_monthly']
merged_file_path = config['merged_file_path']
topic_threshold_percentage = config['topic_threshold_percentage']

#%% Process the topics
print("Processing topics...")
processed_df = process_topics(topic_input_path, topic_output_path, topics_to_keep, topic_threshold_percentage)
print(f"Processed DataFrame columns: {processed_df.columns}")

# Ensure 'permco' is a string in all DataFrames
processed_df['permco'] = processed_df['permco'].astype(str)

# Extract unique permcos from processed_df
permcos = set(processed_df['permco'].unique())
print(f"Number of unique permcos in processed_df: {len(permcos)}")

# Process the CRSP daily data
print("Processing CRSP/Daily data...")
chunksize = 10 ** 6
daily_data = []
for chunk_num, chunk in enumerate(pd.read_csv(file_path_crsp_daily, chunksize=chunksize), start=1):
    chunk['permco'] = chunk['permco'].astype(str)
    chunk['gvkey'] = chunk['gvkey'].astype(str)
    filtered_chunk = chunk[chunk['permco'].isin(permcos)]
    if not filtered_chunk.empty:
        daily_data.append(filtered_chunk)
    print(f"Processed chunk {chunk_num}: {len(chunk)} rows, {len(filtered_chunk)} matching rows.")

if daily_data:
    df_crsp_daily = pd.concat(daily_data, ignore_index=True)
    print(f"Total rows after concatenating daily data: {len(df_crsp_daily)}")
else:
    print("No matching data found in CRSP Daily data.")
    sys.exit()

# Ensure 'date' in CRSP daily is in datetime format
df_crsp_daily['date'] = pd.to_datetime(df_crsp_daily['date'], errors='coerce')
df_crsp_daily = df_crsp_daily[df_crsp_daily['date'].notna()]
print(f"CRSP Daily data after removing NaN dates: {len(df_crsp_daily)} rows.")

# Convert 'gvkey' in df_crsp_daily to numeric and handle missing values
df_crsp_daily['gvkey'] = pd.to_numeric(df_crsp_daily['gvkey'], errors='coerce')

missing_gvkey_daily = df_crsp_daily['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in df_crsp_daily before filling: {missing_gvkey_daily}")

if missing_gvkey_daily > 0:
    print("Filling missing 'gvkey' values in df_crsp_daily based on the most frequent 'gvkey' per 'permco'...")
    # Compute the most frequent gvkey per permco
    most_common_gvkey_per_permco_daily = df_crsp_daily.groupby('permco')['gvkey'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    
    # Map the most frequent gvkey back to df_crsp_daily
    df_crsp_daily['gvkey'] = df_crsp_daily.apply(
        lambda row: most_common_gvkey_per_permco_daily[row['permco']] if pd.isna(row['gvkey']) else row['gvkey'],
        axis=1
    )
    
    # Check missing 'gvkey's again
    missing_gvkey_daily = df_crsp_daily['gvkey'].isna().sum()
    print(f"Number of missing 'gvkey' values in df_crsp_daily after filling: {missing_gvkey_daily}")

# Remove rows with missing 'gvkey' in df_crsp_daily
df_crsp_daily = df_crsp_daily[df_crsp_daily['gvkey'].notna()]
df_crsp_daily['gvkey'] = df_crsp_daily['gvkey'].astype(int)
print(f"CRSP Daily data after removing rows with missing 'gvkey': {len(df_crsp_daily)} rows.")

# Convert 'date' in processed_df to 'call_date' in datetime
processed_df['call_date'] = pd.to_datetime(processed_df['date'], utc=True, errors='coerce')
processed_df = processed_df.drop(columns=['date'])
print("Converted 'date' to 'call_date' in processed_df.")

# Remove rows with missing 'call_date' or 'permco'
initial_len = len(processed_df)
processed_df = processed_df[processed_df['call_date'].notna() & processed_df['permco'].notna()]
print(f"Removed {initial_len - len(processed_df)} rows with missing 'call_date' or 'permco'.")

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
print(f"Number of rows after initial merge: {len(processed_df)}")

# Handle missing 'gvkey' values in processed_df
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in processed_df after merging: {missing_gvkey}")

if missing_gvkey > 0:
    print("Filling missing 'gvkey' values in processed_df based on the most frequent 'gvkey' per 'permco'...")
    # Use the most common gvkey per permco from df_crsp_daily
    processed_df['gvkey'] = processed_df.apply(
        lambda row: most_common_gvkey_per_permco_daily[row['permco']] if pd.isna(row['gvkey']) else row['gvkey'],
        axis=1
    )

# Check for remaining missing 'gvkey's
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in processed_df after filling: {missing_gvkey}")

# Remove rows with missing 'gvkey'
processed_df = processed_df[processed_df['gvkey'].notna()]
processed_df['gvkey'] = processed_df['gvkey'].astype(int)
print(f"processed_df after removing rows with missing 'gvkey': {len(processed_df)} rows.")

# Extract updated set of gvkeys
gvkeys = set(processed_df['gvkey'].unique())
print(f"Number of unique gvkeys in processed_df after adding 'gvkey': {len(gvkeys)}")

#%% Process the CRSP Monthly data
print("Processing CRSP/Monthly data...")
chunksize = 10 ** 6
monthly_data = []
for chunk_num, chunk in enumerate(pd.read_csv(file_path_crsp_monthly, chunksize=chunksize), start=1):
    chunk['gvkey'] = pd.to_numeric(chunk['gvkey'], errors='coerce')
    chunk['permco'] = chunk['permco'].astype(str)
    chunk = chunk[chunk['gvkey'].notna()]
    chunk['gvkey'] = chunk['gvkey'].astype(int)
    matching_gvkeys = chunk['gvkey'].isin(gvkeys)
    num_matching = matching_gvkeys.sum()
    print(f"Chunk {chunk_num}: {len(chunk)} rows, Matching gvkeys: {num_matching}")
    if num_matching > 0:
        filtered_chunk = chunk[matching_gvkeys]
        monthly_data.append(filtered_chunk)

if monthly_data:
    df_crsp_monthly = pd.concat(monthly_data, ignore_index=True)
    print(f"Total rows after concatenating monthly data: {len(df_crsp_monthly)}")
else:
    print("No matching data found in CRSP Monthly data.")
    sys.exit()

# Ensure 'datadate' in df_crsp_monthly is in datetime format and remove NaNs
df_crsp_monthly = df_crsp_monthly[["datadate", "epsfxq", "gvkey", "siccd", "permco"]]
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
df_crsp_monthly['datadate'] = df_crsp_monthly['datadate'].dt.tz_localize(None)
print(f"CRSP Monthly data after removing NaN 'datadate's: {len(df_crsp_monthly)} rows.")

# Handle missing 'siccd' values in df_crsp_monthly
missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
print(f"Number of missing 'siccd' values in df_crsp_monthly before filling: {missing_siccd_monthly}")

if missing_siccd_monthly > 0:
    print("Filling missing 'siccd' values in df_crsp_monthly based on the most frequent 'siccd' per 'permco'...")
    # Compute the most frequent siccd per permco
    most_common_siccd_per_permco_monthly = df_crsp_monthly.groupby('permco')['siccd'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    
    # Map the most frequent siccd back to df_crsp_monthly
    df_crsp_monthly['siccd'] = df_crsp_monthly.apply(
        lambda row: most_common_siccd_per_permco_monthly[row['permco']] if pd.isna(row['siccd']) else row['siccd'],
        axis=1
    )
    
    # Check missing 'siccd's again
    missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
    print(f"Number of missing 'siccd' values in df_crsp_monthly after filling: {missing_siccd_monthly}")

# Convert 'siccd' to integer (if possible)
df_crsp_monthly['siccd'] = pd.to_numeric(df_crsp_monthly['siccd'], errors='coerce').astype('Int64')

# Ensure 'call_date' and 'datadate' are in datetime format and timezone naive
processed_df['call_date'] = pd.to_datetime(processed_df['call_date'], errors='coerce').dt.tz_localize(None)
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce').dt.tz_localize(None)

# Remove rows with missing dates
processed_df = processed_df[processed_df['call_date'].notna()]
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
print(f"After removing rows with missing dates: processed_df={len(processed_df)}, df_crsp_monthly={len(df_crsp_monthly)}")

# Create 'epsfxq_next'
print("Creating 'epsfxq_next' in df_crsp_monthly...")

def get_epsfxq_next(group):
    group = group.sort_values('datadate').reset_index(drop=True)
    group['epsfxq_next'] = group['epsfxq'].shift(-1)
    # Only keep 'epsfxq_next' where the next 'datadate' is approximately 90 days after current 'datadate'
    days_diff = group['datadate'].diff(-1).abs().dt.days
    group['epsfxq_next'] = np.where(
        (days_diff >= 60) & (days_diff <= 120),
        group['epsfxq_next'],
        np.nan
    )
    return group

# Addressing DeprecationWarning by excluding grouping columns explicitly
df_crsp_monthly = df_crsp_monthly.groupby('gvkey', group_keys=False).apply(get_epsfxq_next).reset_index(drop=True)
print("Columns after adding 'epsfxq_next':", df_crsp_monthly.columns.tolist())

# Prepare df_crsp_monthly_for_merge
df_crsp_monthly_for_merge = df_crsp_monthly[['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd']]

# Remove rows with missing merge keys
print("Dropping rows with missing 'gvkey' or 'datadate' in monthly data...")
processed_df = processed_df.dropna(subset=['gvkey', 'call_date'])
df_crsp_monthly_for_merge = df_crsp_monthly_for_merge.dropna(subset=['gvkey', 'datadate'])
print(f"After dropping missing merge keys: processed_df={len(processed_df)}, df_crsp_monthly_for_merge={len(df_crsp_monthly_for_merge)}")

# Ensure data types are correct
processed_df['gvkey'] = processed_df['gvkey'].astype(int)
df_crsp_monthly_for_merge['gvkey'] = df_crsp_monthly_for_merge['gvkey'].astype(int)
processed_df['call_date'] = pd.to_datetime(processed_df['call_date'])
df_crsp_monthly_for_merge['datadate'] = pd.to_datetime(df_crsp_monthly_for_merge['datadate'])

# Remove duplicates
processed_df = processed_df.drop_duplicates(subset=['gvkey', 'call_date'])
df_crsp_monthly_for_merge = df_crsp_monthly_for_merge.drop_duplicates(subset=['gvkey', 'datadate'])
print(f"After removing duplicates: processed_df={len(processed_df)}, df_crsp_monthly_for_merge={len(df_crsp_monthly_for_merge)}")

# Sort DataFrames explicitly by 'gvkey' and date columns
print("Sorting DataFrames...")
processed_df = processed_df.sort_values(by=['gvkey', 'call_date'], ascending=[True, True]).reset_index(drop=True)
df_crsp_monthly_for_merge = df_crsp_monthly_for_merge.sort_values(by=['gvkey', 'datadate'], ascending=[True, True]).reset_index(drop=True)

# Re-validate sorting
if not is_sorted_within_group(processed_df, 'gvkey', 'call_date'):
    print("Error: 'call_date' is not sorted within 'gvkey' in 'processed_df'.")
    sys.exit()

if not is_sorted_within_group(df_crsp_monthly_for_merge, 'gvkey', 'datadate'):
    print("Error: 'datadate' is not sorted within 'gvkey' in 'df_crsp_monthly_for_merge'.")
    sys.exit()

# Perform the merge_asof
print("Performing merge_asof...")
merged_df = pd.merge_asof(
    processed_df,
    df_crsp_monthly_for_merge,
    left_on='call_date',
    right_on='datadate',
    by='gvkey',
    direction='backward',
    allow_exact_matches=True
)
print(f"Number of rows in merged_df after merge_asof: {len(merged_df)}")

# Handle missing 'siccd' values in merged_df
num_nan_siccd = merged_df['siccd'].isna().sum()
print(f"Number of rows with NaN 'siccd': {num_nan_siccd}")

# Fill missing 'siccd' values based on 'permco' using df_crsp_monthly
print("Filling missing 'siccd' values based on the most frequent 'siccd' per 'permco' from df_crsp_monthly...")
siccd_mapping = df_crsp_monthly.groupby('permco')['siccd'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Ensure 'permco' is present in 'merged_df'
if 'permco' not in merged_df.columns:
    print("Error: 'permco' column is missing in 'merged_df'.")
else:
    merged_df['permco'] = merged_df['permco'].astype(str)
    merged_df['siccd'] = merged_df.apply(
        lambda row: siccd_mapping[row['permco']] if pd.isna(row['siccd']) else row['siccd'],
        axis=1
    )

# Check how many NaNs are left in 'siccd'
num_nan_siccd = merged_df['siccd'].isna().sum()
print(f"Number of rows with NaN 'siccd' after filling from df_crsp_monthly: {num_nan_siccd}")

# Proceed without dropping rows
if num_nan_siccd > 0:
    print(f"Proceeding with {num_nan_siccd} rows with NaN 'siccd'. Similarity to industry average will be NaN for these rows.")

# Optionally, rename 'datadate' to 'fiscal_period_end' for clarity
merged_df.rename(columns={'datadate': 'fiscal_period_end'}, inplace=True)

# Now that 'siccd' is available, compute similarities including similarity to industry average
print("Computing similarities to overall and industry averages...")
# Determine the number of topics
num_topics = merged_df['filtered_topics'].apply(lambda x: max(x) if x else 0).max() + 1
print(f"Number of topics determined: {num_topics}")

# Ensure 'filtered_topics' is evaluated as lists
merged_df['filtered_topics'] = merged_df['filtered_topics'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Ensure 'call_date' is in datetime format
merged_df['call_date'] = pd.to_datetime(merged_df['call_date'])

# Convert 'siccd' to integer (if possible)
merged_df['siccd'] = merged_df['siccd'].astype(float).astype('Int64')

# Compute similarities
similarity_df = compute_similarity_to_average(merged_df, num_topics)

# Merge similarities back into merged_df
merged_df = merged_df.merge(similarity_df, on='call_id', how='left')
print("Merged similarity measures into merged_df.")

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
merged_df = merged_df.drop(columns=['date'], errors='ignore')
print(f"Number of rows after merging with CRSP/Daily data: {len(merged_df)}")

# Convert 'ret' to numeric, handling non-numeric values
def clean_ret(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

merged_df['ret'] = merged_df['ret'].apply(clean_ret)
print(f"\nNumber of NaNs in 'ret' after cleaning: {merged_df['ret'].isna().sum()}")

# Proceed with computing future returns
print("Computing future returns...")
df_crsp_daily = df_crsp_daily.sort_values(['permco', 'date']).reset_index(drop=True)

def compute_future_returns(group):
    group = group.sort_values('date').reset_index(drop=True)
    n = len(group)
    ret_values = group['ret'].values
    ret_next_day = np.full(n, np.nan)
    ret_5_days = np.full(n, np.nan)
    ret_20_days = np.full(n, np.nan)
    ret_60_days = np.full(n, np.nan)
    
    for i in range(n):
        # Next day return
        if i + 1 < n and not np.isnan(ret_values[i+1]):
            ret_next_day[i] = ret_values[i+1]
        
        # 5 days return
        if i + 5 < n and not np.isnan(ret_values[i+1:i+6]).any():
            ret_5_days[i] = np.prod(1 + ret_values[i+1:i+6]) - 1
        
        # 20 days return
        if i + 20 < n and not np.isnan(ret_values[i+1:i+21]).any():
            ret_20_days[i] = np.prod(1 + ret_values[i+1:i+21]) - 1
        
        # 60 days return
        if i + 60 < n and not np.isnan(ret_values[i+1:i+61]).any():
            ret_60_days[i] = np.prod(1 + ret_values[i+1:i+61]) - 1
    
    group['ret_next_day'] = ret_next_day
    group['ret_5_days'] = ret_5_days
    group['ret_20_days'] = ret_20_days
    group['ret_60_days'] = ret_60_days
    return group

df_crsp_daily = df_crsp_daily.groupby('permco', group_keys=False).apply(compute_future_returns).reset_index(drop=True)
print("Completed computation of future returns.")

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
merged_df = merged_df.drop(columns=['date'], errors='ignore')
print(f"Number of rows after merging future returns: {len(merged_df)}")

# Rearrange columns to include 'epsfxq_next' and similarity measures
desired_columns = [
    'gvkey', 'permco', 'siccd', 'call_id', 'call_date', 'fiscal_period_end',
    'filtered_topics', 'filtered_texts', 'prc', 'shrout', 'vol', 'ret',
    'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days',
    'epsfxq', 'epsfxq_next',
    'similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average'
]

# Check if all desired columns are present
missing_cols = set(desired_columns) - set(merged_df.columns)
if missing_cols:
    print(f"Warning: The following expected columns are missing in merged_df and will be excluded from the final DataFrame: {missing_cols}")

# Select only the columns that exist
final_columns = [col for col in desired_columns if col in merged_df.columns]
merged_df = merged_df[final_columns]

# Sort the final DataFrame by 'gvkey' and 'call_date' in ascending order
print("Sorting the final DataFrame by 'gvkey' and 'call_date'...")
merged_df = merged_df.sort_values(by=['gvkey', 'call_date']).reset_index(drop=True)

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
print(f"Final merged DataFrame saved to {merged_file_path}")

# Calculate the means of the similarity measures
print("Calculating means of similarity measures...")
for similarity_col in ['similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']:
    if similarity_col in merged_df.columns:
        mean_value = merged_df[similarity_col].mean()
        print(f"Mean of {similarity_col}: {mean_value}")
    else:
        print(f"Column '{similarity_col}' not found in merged_df.")

print("Dataset creation completed successfully.")
