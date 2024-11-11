# create_final_dataset.py

import json
import pandas as pd
import numpy as np
import sys
import warnings
import ast
from utils import process_topics, compute_similarity_to_average

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

# Process the topics
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

# Convert 'date' in processed_df to 'call_date' in datetime
processed_df['call_date'] = pd.to_datetime(processed_df['date'], utc=True, errors='coerce')
processed_df = processed_df.drop(columns=['date'])
print("Converted 'date' to 'call_date' in processed_df.")

# Remove rows with missing 'call_date' or 'permco'
initial_len = len(processed_df)
processed_df = processed_df[processed_df['call_date'].notna() & processed_df['permco'].notna()]
print(f"Removed {initial_len - len(processed_df)} rows with missing 'call_date' or 'permco'.")

# Convert 'call_date' to New York time and remove timezone information
processed_df['call_date'] = processed_df['call_date'].dt.tz_convert('America/New_York')
processed_df['call_date_with_time'] = processed_df['call_date'].dt.tz_localize(None)
print("Converted 'call_date' to New York time and stored in 'call_date_with_time'.")

# Normalize 'call_date' to remove time component for merging
processed_df['call_date'] = processed_df['call_date_with_time'].dt.normalize()
print("Normalized 'call_date' to remove time component for merging.")

# Merge processed_df with df_crsp_daily to get 'gvkey' into processed_df
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

# After merging processed_df with df_crsp_daily
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values in processed_df after merging: {missing_gvkey}")

if missing_gvkey > 0:
    print("Filling missing 'gvkey' values in processed_df based on the most frequent 'gvkey' per 'permco'...")
    # Create mapping from 'permco' to most frequent 'gvkey' in df_crsp_daily
    gvkey_mapping = df_crsp_daily.groupby('permco')['gvkey'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    # Fill missing 'gvkey's in processed_df
    processed_df['gvkey'] = processed_df.apply(
        lambda row: gvkey_mapping.get(row['permco'], np.nan) if pd.isna(row['gvkey']) else row['gvkey'],
        axis=1
    )
    # Check how many 'gvkey's are still missing
    missing_gvkey = processed_df['gvkey'].isna().sum()
    print(f"Number of missing 'gvkey' values after filling: {missing_gvkey}")

# Ensure 'gvkey' is numeric
processed_df['gvkey'] = pd.to_numeric(processed_df['gvkey'], errors='coerce').astype('Int64')
df_crsp_daily['gvkey'] = pd.to_numeric(df_crsp_daily['gvkey'], errors='coerce').astype('Int64')

# Remove any remaining rows with missing 'gvkey' in processed_df
processed_df = processed_df.dropna(subset=['gvkey'])
processed_df['gvkey'] = processed_df['gvkey'].astype(int)

# Extract updated set of gvkeys
gvkeys = set(processed_df['gvkey'].unique())
print(f"Number of unique gvkeys in processed_df after adding 'gvkey': {len(gvkeys)}")

# Process the CRSP Monthly data
print("Processing CRSP/Monthly data...")
chunksize = 10 ** 6
monthly_data = []
for chunk_num, chunk in enumerate(pd.read_csv(file_path_crsp_monthly, chunksize=chunksize), start=1):
    chunk['gvkey'] = pd.to_numeric(chunk['gvkey'], errors='coerce').astype('Int64')
    chunk['permco'] = chunk['permco'].astype(str)
    filtered_chunk = chunk[chunk['gvkey'].isin(gvkeys)]
    if not filtered_chunk.empty:
        monthly_data.append(filtered_chunk)
    print(f"Processed chunk {chunk_num}: {len(chunk)} rows, {len(filtered_chunk)} matching rows.")

if monthly_data:
    df_crsp_monthly = pd.concat(monthly_data, ignore_index=True)
    print(f"Total rows after concatenating monthly data: {len(df_crsp_monthly)}")
else:
    print("No matching data found in CRSP Monthly data.")
    # Create an empty DataFrame with the necessary columns
    df_crsp_monthly = pd.DataFrame(columns=['datadate', 'epsfxq', 'gvkey', 'siccd', 'permco'])

# Ensure 'datadate' in df_crsp_monthly is in datetime format and remove NaNs
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
print(f"CRSP Monthly data after removing NaN 'datadate's: {len(df_crsp_monthly)} rows.")

# Handle missing 'siccd' values in df_crsp_monthly
missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
print(f"Number of missing 'siccd' values in df_crsp_monthly before filling: {missing_siccd_monthly}")

if missing_siccd_monthly > 0:
    print("Filling missing 'siccd' values in df_crsp_monthly based on the most frequent 'siccd' per 'permco'...")
    # Compute the most frequent siccd per permco
    most_common_siccd_per_permco_monthly = df_crsp_monthly.groupby('permco')['siccd'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    # Map the most frequent siccd back to df_crsp_monthly
    df_crsp_monthly['siccd'] = df_crsp_monthly.apply(
        lambda row: most_common_siccd_per_permco_monthly.get(row['permco'], np.nan) if pd.isna(row['siccd']) else row['siccd'],
        axis=1
    )
    # Check missing 'siccd's again
    missing_siccd_monthly = df_crsp_monthly['siccd'].isna().sum()
    print(f"Number of missing 'siccd' values in df_crsp_monthly after filling: {missing_siccd_monthly}")

# Convert 'siccd' to integer (if possible)
df_crsp_monthly['siccd'] = pd.to_numeric(df_crsp_monthly['siccd'], errors='coerce').astype('Int64')

# Ensure 'datadate' is timezone naive
df_crsp_monthly['datadate'] = df_crsp_monthly['datadate'].dt.tz_localize(None)

# Remove duplicates
processed_df = processed_df.drop_duplicates(subset=['gvkey', 'call_date'])
df_crsp_monthly = df_crsp_monthly.drop_duplicates(subset=['gvkey', 'datadate'])
print(f"After removing duplicates: processed_df={len(processed_df)}, df_crsp_monthly={len(df_crsp_monthly)}")

# Convert 'call_date' and 'datadate' to datetime64[ns] and ensure they are timezone naive
processed_df['call_date'] = pd.to_datetime(processed_df['call_date']).dt.tz_localize(None)
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate']).dt.tz_localize(None)

# Drop rows with missing merge keys
processed_df.dropna(subset=['gvkey', 'call_date'], inplace=True)
df_crsp_monthly.dropna(subset=['gvkey', 'datadate'], inplace=True)

# Sort DataFrames
processed_df.sort_values(by=['gvkey', 'call_date'], inplace=True)
df_crsp_monthly.sort_values(by=['gvkey', 'datadate'], inplace=True)

# Reset indices
processed_df.reset_index(drop=True, inplace=True)
df_crsp_monthly.reset_index(drop=True, inplace=True)

# Create 'epsfxq_next' in df_crsp_monthly
print("Creating 'epsfxq_next' in df_crsp_monthly...")

def get_epsfxq_next(group):
    group = group.sort_values('datadate').reset_index(drop=True)
    group['epsfxq_next'] = group['epsfxq'].shift(-1)
    group['datadate_next'] = group['datadate'].shift(-1)
    
    # Calculate the difference in days between current and next datadate
    group['days_diff'] = (group['datadate_next'] - group['datadate']).dt.days
    
    # Only assign epsfxq_next if the next datadate is approximately one quarter later
    group['epsfxq_next'] = np.where(
        (group['days_diff'] >= 60) & (group['days_diff'] <= 120),
        group['epsfxq_next'],
        np.nan
    )
    
    # Drop helper columns
    group = group.drop(columns=['datadate_next', 'days_diff'])
    return group

df_crsp_monthly = df_crsp_monthly.groupby('gvkey', group_keys=False).apply(get_epsfxq_next)

print("Completed 'epsfxq_next' computation with missing quarters adjusted.")

# **Select necessary columns from processed_df**
processed_df_columns = ['call_id', 'gvkey', 'call_date', 'call_date_with_time', 'permco', 'filtered_topics', 'filtered_texts']
processed_df = processed_df[processed_df_columns]

# **Select necessary columns from df_crsp_monthly**
df_crsp_monthly_columns = ['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd', 'permco']
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly_columns]

# Merge processed_df with df_crsp_monthly
print("Performing custom merge between processed_df and df_crsp_monthly...")
# Step 1: Perform a full merge on 'gvkey' to create all possible combinations.
temp_merged = pd.merge(
    processed_df.drop(columns=['call_date_with_time']),  # Exclude 'call_date_with_time' to avoid duplication
    df_crsp_monthly[['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd']],
    on='gvkey'
)

# Step 2: Filter out rows where 'datadate' is after 'call_date' (keeping only those before or exactly on).
temp_merged = temp_merged[temp_merged['datadate'] <= temp_merged['call_date']]

# Step 3: For each 'gvkey' and 'call_date' group, keep only the row with the most recent 'datadate'.
merged_df = temp_merged.sort_values(['gvkey', 'call_date', 'datadate']).drop_duplicates(['gvkey', 'call_date'], keep='last')

print(f"Number of rows in merged_df after merging: {len(merged_df)}")

# Handle missing 'siccd' values in merged_df
num_nan_siccd = merged_df['siccd'].isna().sum()
print(f"Number of rows with NaN 'siccd': {num_nan_siccd}")

# Fill missing 'siccd' values based on 'permco' using df_crsp_monthly
print("Filling missing 'siccd' values based on the most frequent 'siccd' per 'permco' from df_crsp_monthly...")
siccd_mapping = df_crsp_monthly.groupby('permco')['siccd'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
)

# Ensure 'permco' is present in 'merged_df'
if 'permco' not in merged_df.columns:
    print("Error: 'permco' column is missing in 'merged_df'.")
else:
    merged_df['siccd'] = merged_df.apply(
        lambda row: siccd_mapping.get(row['permco'], row['siccd']) if pd.isna(row['siccd']) else row['siccd'],
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
print("Computing similarities to overall, industry and firm-specific averages...")
# Determine the number of topics
try:
    num_topics = merged_df['filtered_topics'].apply(lambda x: max(x) if isinstance(x, list) and x else 0).max() + 1
except Exception as e:
    print(f"Error determining number of topics: {e}")
    num_topics = 0  # Fallback or handle accordingly
print(f"Number of topics determined: {num_topics}")

# Ensure 'filtered_topics' is evaluated as lists
merged_df['filtered_topics'] = merged_df['filtered_topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert 'siccd' to integer (if possible)
merged_df['siccd'] = merged_df['siccd'].astype(float).astype('Int64')

# Compute similarities
similarity_df = compute_similarity_to_average(merged_df, num_topics)
print("Computed similarity measures.")

# Merge similarities back into merged_df
if 'call_id' in similarity_df.columns and 'call_id' in merged_df.columns:
    merged_df = merged_df.merge(similarity_df, on='call_id', how='left')
    print("Merged similarity measures into merged_df.")
else:
    print("Warning: 'call_id' column not found in similarity_df or merged_df. Skipping similarity merge.")

# Merge with CRSP daily data to get financial variables
print("Merging with CRSP/Daily data to get financial variables...")
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
print(f"Number of NaNs in 'ret' after cleaning: {merged_df['ret'].isna().sum()}")

# Compute future returns
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

        # 5 days return (starting one day after ret_next_day)
        if i + 6 < n and not np.isnan(ret_values[i+2:i+7]).any():
            ret_5_days[i] = np.prod(1 + ret_values[i+2:i+7]) - 1

        # 20 days return
        if i + 21 < n and not np.isnan(ret_values[i+2:i+22]).any():
            ret_20_days[i] = np.prod(1 + ret_values[i+2:i+22]) - 1

        # 60 days return
        if i + 61 < n and not np.isnan(ret_values[i+2:i+62]).any():
            ret_60_days[i] = np.prod(1 + ret_values[i+2:i+62]) - 1

    group['ret_next_day'] = ret_next_day
    group['ret_5_days'] = ret_5_days
    group['ret_20_days'] = ret_20_days
    group['ret_60_days'] = ret_60_days
    return group

df_crsp_daily = df_crsp_daily.groupby('permco', group_keys=False).apply(compute_future_returns).reset_index(drop=True)
print("Completed computation of future returns.")

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

# Compute market returns
print("Computing market returns...")
market_returns = df_crsp_daily.groupby('date')['ret'].mean().reset_index()
market_returns.rename(columns={'ret': 'market_ret'}, inplace=True)

# Merge market returns into merged_df
merged_df = pd.merge(
    merged_df,
    market_returns,
    left_on='call_date',
    right_on='date',
    how='left'
)
merged_df = merged_df.drop(columns=['date'], errors='ignore')
print("Merged market returns into merged_df.")

# Compute excess returns
print("Computing excess returns...")
merged_df['excess_ret'] = merged_df['ret'] - merged_df['market_ret']
merged_df['excess_ret_next_day'] = merged_df['ret_next_day'] - merged_df['market_ret']
merged_df['excess_ret_5_days'] = merged_df['ret_5_days'] - merged_df['market_ret']
merged_df['excess_ret_20_days'] = merged_df['ret_20_days'] - merged_df['market_ret']
merged_df['excess_ret_60_days'] = merged_df['ret_60_days'] - merged_df['market_ret']

# Shift future returns one day later
print("Shifting future returns one day later...")
def shift_future_returns(group):
    group = group.sort_values('call_date').reset_index(drop=True)
    group['excess_ret_5_days'] = group['excess_ret_5_days'].shift(-1)
    group['excess_ret_20_days'] = group['excess_ret_20_days'].shift(-1)
    group['excess_ret_60_days'] = group['excess_ret_60_days'].shift(-1)
    return group

merged_df = merged_df.groupby('permco', group_keys=False).apply(shift_future_returns).reset_index(drop=True)
print("Future returns shifted.")

# **Restore 'call_date' with time component**
print("Restoring 'call_date' with time component from 'processed_df'...")

# Drop 'call_date_with_time' from merged_df if exists
merged_df = merged_df.drop(columns=['call_date_with_time'], errors='ignore')

# Merge 'call_date_with_time' back into merged_df using 'call_id'
merged_df = pd.merge(
    merged_df,
    processed_df[['call_id', 'call_date_with_time']],
    on='call_id',
    how='left'
)

# Replace 'call_date' in merged_df with 'call_date_with_time'
merged_df['call_date'] = merged_df['call_date_with_time']

# Drop 'call_date_with_time' column
merged_df = merged_df.drop(columns=['call_date_with_time'])
print("Restored 'call_date' with time component in merged_df.")

# Final DataFrame Preparation and Saving
print("Finalizing the merged DataFrame...")
desired_columns = [
    'filtered_topics', 'filtered_texts', 'call_id', 'permco', 'call_date', 'gvkey',
    'fiscal_period_end', 'epsfxq', 'epsfxq_next', 'siccd',
    'similarity_to_overall_average', 'similarity_to_industry_average',
    'similarity_to_company_average', 'prc', 'shrout', 'ret', 'vol',
    'market_ret', 'excess_ret', 'excess_ret_next_day', 'excess_ret_5_days',
    'excess_ret_20_days', 'excess_ret_60_days'
]

# Check if all desired columns are present
missing_cols = set(desired_columns) - set(merged_df.columns)
if missing_cols:
    print(f"Warning: The following expected columns are missing in merged_df and will be excluded from the final DataFrame: {missing_cols}")

#take absolute values for the prc column
merged_df['prc'] = merged_df['prc'].abs()

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
