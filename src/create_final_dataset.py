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

# Process the topics
print("Processing topics...")
processed_df = process_topics(topic_input_path, topic_output_path, topics_to_keep)
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

# Merge processed_df with df_crsp_daily to get 'gvkey' into processed_df
print("Merging processed_df with df_crsp_daily to get 'gvkey'...")
processed_df = pd.merge(
    processed_df,
    df_crsp_daily[['permco', 'date', 'gvkey']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
processed_df = processed_df.drop(columns=['date'])

# Handle missing 'gvkey' values
missing_gvkey = processed_df['gvkey'].isna().sum()
print(f"Number of missing 'gvkey' values after merging with df_crsp_daily: {missing_gvkey}")

# Remove rows with missing 'gvkey' if necessary
processed_df = processed_df[processed_df['gvkey'].notna()]
processed_df['gvkey'] = processed_df['gvkey'].astype(str)

# Process the CRSP monthly data
print("Processing CRSP/Monthly data...")
monthly_data = []
for chunk in pd.read_csv(file_path_crsp_monthly, chunksize=chunksize):
    chunk['gvkey'] = chunk['gvkey'].astype(str)
    filtered_chunk = chunk[chunk['gvkey'].isin(processed_df['gvkey'].unique())]
    if not filtered_chunk.empty:
        monthly_data.append(filtered_chunk)

df_crsp_monthly = pd.concat(monthly_data, ignore_index=True)

# Ensure 'datadate' in df_crsp_monthly is in datetime format and remove NaNs
df_crsp_monthly = df_crsp_monthly[["datadate", "epsfxq", "gvkey", "siccd"]]
df_crsp_monthly['datadate'] = pd.to_datetime(df_crsp_monthly['datadate'], errors='coerce')
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly['datadate'].notna()]
df_crsp_monthly['gvkey'] = df_crsp_monthly['gvkey'].astype(str)

# Create 'epsfxq_next' by shifting 'epsfxq' within each 'gvkey'
df_crsp_monthly = df_crsp_monthly.sort_values(by=['gvkey', 'datadate']).reset_index(drop=True)
df_crsp_monthly['epsfxq_next'] = df_crsp_monthly.groupby('gvkey')['epsfxq'].shift(-1)

# Ensure both DataFrames are sorted properly
print("Ensuring both DataFrames are properly sorted...")
processed_df = processed_df.sort_values(by=['call_date', 'gvkey'], ascending=[True, True]).reset_index(drop=True)
df_crsp_monthly = df_crsp_monthly.sort_values(by=['datadate', 'gvkey'], ascending=[True, True]).reset_index(drop=True)

# Verify global sorting
print("Checking if 'call_date' is globally sorted in processed_df...")
is_call_date_sorted = processed_df['call_date'].is_monotonic_increasing
print(f"Is 'call_date' globally sorted? {is_call_date_sorted}")

print("Checking if 'datadate' is globally sorted in df_crsp_monthly...")
is_datadate_sorted = df_crsp_monthly['datadate'].is_monotonic_increasing
print(f"Is 'datadate' globally sorted? {is_datadate_sorted}")

# Merge using merge_asof with direction='backward' to get 'epsfxq', 'epsfxq_next', and 'siccd' using 'gvkey'
print("Merging processed_df with df_crsp_monthly using 'gvkey' and merge_asof...")
merged_df = pd.merge_asof(
    processed_df,
    df_crsp_monthly[['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd']],
    left_on='call_date',
    right_on='datadate',
    by='gvkey',
    direction='backward',
    allow_exact_matches=True
)

# Optionally, rename 'datadate' to 'fiscal_period_end' for clarity
merged_df.rename(columns={'datadate': 'fiscal_period_end'}, inplace=True)

# Now that 'siccd' is available, compute similarities including similarity to industry average
print("Computing similarities to overall and industry averages...")
num_topics = merged_df['filtered_topics'].apply(lambda x: max(x) if x else 0).max() + 1

# Ensure 'filtered_topics' is evaluated as lists
merged_df['filtered_topics'] = merged_df['filtered_topics'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Ensure 'siccd' is not null
merged_df = merged_df[merged_df['siccd'].notna()]

# Convert 'siccd' to integer (if needed)
merged_df['siccd'] = merged_df['siccd'].astype(int)

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
    except ValueError:
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
merged_df = merged_df.drop(columns=['date', 'topics', 'text', 'consistent'])

# Rearrange columns to include 'epsfxq_next' and similarity measures
merged_df = merged_df[['gvkey', 'permco', 'siccd', 'call_id', 'call_date', 'fiscal_period_end', 'filtered_topics', 'filtered_texts',
                       'prc', 'shrout', 'vol', 'ret', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days',
                       'epsfxq', 'epsfxq_next', 'similarity_to_overall_average', 'similarity_to_industry_average']]

# Fill NaNs in 'siccd' based on 'gvkey'
print("Filling NaNs in 'siccd' column based on 'gvkey'...")
merged_df['siccd'] = merged_df.groupby('gvkey')['siccd'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Sort the final DataFrame by 'gvkey' and 'call_date' in ascending order
print("Sorting the final DataFrame by 'gvkey' and 'call_date'...")
merged_df = merged_df.sort_values(by=['gvkey', 'call_date'], ascending=[True, True]).reset_index(drop=True)

# Save the final merged DataFrame
print("Saving the final merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
print(f"Final merged DataFrame saved to {merged_file_path}")
