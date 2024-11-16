# create_final_dataset.py

import json
import pandas as pd
import numpy as np
import sys
import warnings
import ast
from utils import process_topics, compute_similarity_to_average

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

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

# -------------------------- #
# **Include 'ceo_participates' and CEO/CFO Names**
# -------------------------- #

# Load topics_output.csv to get 'ceo_participates', 'ceo_names', 'cfo_names'
print("Loading topics_output.csv to get 'ceo_participates', 'ceo_names', 'cfo_names'...")
topics_output_df = pd.read_csv(topic_output_path)

# Ensure 'call_id' is present in both DataFrames
if 'call_id' in processed_df.columns and 'call_id' in topics_output_df.columns:
    # Merge 'ceo_participates', 'ceo_names', 'cfo_names' into processed_df
    processed_df = pd.merge(
        processed_df,
        topics_output_df[['call_id', 'ceo_participates', 'ceo_names', 'cfo_names', 'date']],
        on='call_id',
        how='left'
    )
else:
    print("Error: 'call_id' column not found in processed_df or topics_output_df.")
    sys.exit()

# Convert 'ceo_participates' to integer (if not already)
processed_df['ceo_participates'] = processed_df['ceo_participates'].fillna(0).astype(int)
print("Merged 'ceo_participates', 'ceo_names', and 'cfo_names' into processed_df.")

# Convert 'ceo_names' and 'cfo_names' from JSON strings to lists
processed_df['ceo_names'] = processed_df['ceo_names'].apply(lambda x: json.loads(x) if pd.notnull(x) else [])
processed_df['cfo_names'] = processed_df['cfo_names'].apply(lambda x: json.loads(x) if pd.notnull(x) else [])

# Convert 'date' to 'call_date' in datetime
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

# -------------------------- #
# **1. Merge 'market_cap' into processed_df**
# -------------------------- #

# Load CRSP daily data in chunks
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

# Merge processed_df with df_crsp_daily to get 'gvkey' and 'market_cap' into processed_df
print("Merging processed_df on permco and call_date with df_crsp_daily to get 'gvkey' and 'market_cap'...")
processed_df = pd.merge(
    processed_df,
    df_crsp_daily[['permco', 'date', 'gvkey', 'market_cap']],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
processed_df = processed_df.drop(columns=['date'])
print(f"Number of rows after initial merge: {len(processed_df)}")

# Handle missing 'gvkey' values
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

# -------------------------- #
# **2. Handle Missing 'market_cap' Values**
# -------------------------- #

# Check for missing 'market_cap' values
missing_market_cap = processed_df['market_cap'].isna().sum()
print(f"Number of missing 'market_cap' values in processed_df after merging: {missing_market_cap}")

if missing_market_cap > 0:
    print("Filling missing 'market_cap' values in processed_df based on the most frequent 'market_cap' per 'permco'...")
    # Create mapping from 'permco' to most frequent 'market_cap' in df_crsp_daily
    market_cap_mapping = df_crsp_daily.groupby('permco')['market_cap'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    # Fill missing 'market_cap's in processed_df
    processed_df['market_cap'] = processed_df.apply(
        lambda row: market_cap_mapping.get(row['permco'], np.nan) if pd.isna(row['market_cap']) else row['market_cap'],
        axis=1
    )
    # Check how many 'market_cap's are still missing
    missing_market_cap = processed_df['market_cap'].isna().sum()
    print(f"Number of missing 'market_cap' values after filling: {missing_market_cap}")

# Proceeding without dropping rows with missing 'market_cap'
if missing_market_cap > 0:
    print(f"Proceeding with {missing_market_cap} rows with NaN 'market_cap'.")

# Remove any remaining rows with missing 'gvkey' in processed_df
processed_df = processed_df.dropna(subset=['gvkey'])
processed_df['gvkey'] = processed_df['gvkey'].astype(int)

# Extract updated set of gvkeys
gvkeys = set(processed_df['gvkey'].unique())
print(f"Number of unique gvkeys in processed_df after adding 'gvkey': {len(gvkeys)}")

# -------------------------- #
# **3. Process the CRSP Monthly Data**
# -------------------------- #

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
processed_df_columns = ['call_id', 'gvkey', 'call_date', 'call_date_with_time', 'permco', 'filtered_topics', 'filtered_texts', 'market_cap', 'ceo_participates', 'ceo_names', 'cfo_names']  # **Added 'market_cap', 'ceo_participates', 'ceo_names', 'cfo_names'**
processed_df = processed_df[processed_df_columns]

# **Select necessary columns from df_crsp_monthly**
df_crsp_monthly_columns = ['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd', 'permco']
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly_columns]

# Merge processed_df with df_crsp_monthly
print("Performing custom merge between processed_df and df_crsp_monthly...")
# Step 1: Perform a full merge on 'gvkey' to create all possible combinations.
temp_merged = pd.merge(
    processed_df.drop(columns=['call_date_with_time', 'ceo_names', 'cfo_names']),  # Exclude 'call_date_with_time' and names to avoid duplication
    df_crsp_monthly[['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd']],
    on='gvkey',
    how='left'
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

# Remove duplicates in merged_df where fiscal_period_end and gvkey and permco are the same
merged_df = merged_df.drop_duplicates(subset=['fiscal_period_end', 'gvkey', 'permco'])
print(f"Number of rows in merged_df after removing duplicates: {len(merged_df)}")

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

# Compute market returns
print("Computing market returns...")
market_returns = df_crsp_daily.groupby('date')['ret'].mean().reset_index()
market_returns.rename(columns={'ret': 'market_ret'}, inplace=True)

# Merge market returns into df_crsp_daily
df_crsp_daily = pd.merge(
    df_crsp_daily,
    market_returns,
    on='date',
    how='left'
)
print("Merged market returns into df_crsp_daily.")

# Compute future returns
print("Computing future returns...")
df_crsp_daily = df_crsp_daily.sort_values(['permco', 'date']).reset_index(drop=True)

def compute_future_returns(group):
    group = group.sort_values('date').reset_index(drop=True)
    n = len(group)
    ret_values = group['ret'].values
    market_ret_values = group['market_ret'].values

    ret_immediate = np.full(n, np.nan)  # Immediate market reaction: t-1 to t+1
    ret_short_term = np.full(n, np.nan)  # Short-term market reaction: t+2 to t+6
    ret_medium_term = np.full(n, np.nan)  # Medium-term market reaction: t+2 to t+21
    ret_long_term = np.full(n, np.nan)  # Long-term market reaction: t+2 to t+60

    excess_ret_immediate = np.full(n, np.nan)
    excess_ret_short_term = np.full(n, np.nan)
    excess_ret_medium_term = np.full(n, np.nan)
    excess_ret_long_term = np.full(n, np.nan)

    for i in range(n):
        # Immediate market reaction: t-1 to t+1
        if i - 1 >= 0 and i + 1 < n:
            returns = ret_values[i - 1 : i + 2]  # i-1 to i+1 inclusive
            market_returns = market_ret_values[i - 1 : i + 2]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_immediate[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_immediate[i] = ret_immediate[i] - market_return

        # Short-term market reaction: t+2 to t+6
        if i + 6 < n:
            returns = ret_values[i + 2 : i + 7]  # t+2 to t+6 inclusive
            market_returns = market_ret_values[i + 2 : i + 7]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_short_term[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_short_term[i] = ret_short_term[i] - market_return

        # Medium-term market reaction: t+2 to t+21
        if i + 21 < n:
            returns = ret_values[i + 2 : i + 22]  # t+2 to t+21 inclusive
            market_returns = market_ret_values[i + 2 : i + 22]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_medium_term[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_medium_term[i] = ret_medium_term[i] - market_return

        # Long-term market reaction: t+2 to t+60
        if i + 60 < n:
            returns = ret_values[i + 2 : i + 62]  # t+2 to t+61 inclusive
            market_returns = market_ret_values[i + 2 : i + 62]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_long_term[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_long_term[i] = ret_long_term[i] - market_return

    group['ret_immediate'] = ret_immediate
    group['ret_short_term'] = ret_short_term
    group['ret_medium_term'] = ret_medium_term
    group['ret_long_term'] = ret_long_term

    group['excess_ret_immediate'] = excess_ret_immediate
    group['excess_ret_short_term'] = excess_ret_short_term
    group['excess_ret_medium_term'] = excess_ret_medium_term
    group['excess_ret_long_term'] = excess_ret_long_term

    return group

df_crsp_daily = df_crsp_daily.groupby('permco', group_keys=False).apply(compute_future_returns).reset_index(drop=True)
print("Completed computation of future returns.")

# Merge future returns into merged_df
print("Merging future returns into the merged DataFrame...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[['permco', 'date', 'ret', 'prc', 'shrout', 'vol',
                   'ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term',
                   'excess_ret_immediate', 'excess_ret_short_term', 'excess_ret_medium_term', 'excess_ret_long_term',
                   'market_cap']],  # **Added 'market_cap'**
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date'], errors='ignore')
print(f"Number of rows after merging future returns and 'market_cap': {len(merged_df)}")

# Convert 'ret' to numeric, handling non-numeric values
def clean_ret(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

merged_df['ret'] = merged_df['ret'].apply(clean_ret)
print(f"Number of NaNs in 'ret' after cleaning: {merged_df['ret'].isna().sum()}")

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

# -------------------------- #
# **4. Create 'ceo_cfo_change' Dummy Variable**
# -------------------------- #

print("Creating 'ceo_cfo_change' dummy variable...")

# Sort merged_df by 'permco' and 'call_date'
merged_df = merged_df.sort_values(['permco', 'call_date']).reset_index(drop=True)

# Group by 'permco' and compute 'ceo_cfo_change'
def compute_ceo_cfo_change(group):
    group = group.sort_values('call_date').reset_index(drop=True)
    group['ceo_cfo_change'] = 0
    prev_ceo_names = None
    prev_cfo_names = None
    for idx, row in group.iterrows():
        ceo_names = set(row['ceo_names'])
        cfo_names = set(row['cfo_names'])
        if idx > 0:
            if (prev_ceo_names != ceo_names) or (prev_cfo_names != cfo_names):
                group.at[idx, 'ceo_cfo_change'] = 1
        prev_ceo_names = ceo_names
        prev_cfo_names = cfo_names
    return group

merged_df = merged_df.groupby('permco', group_keys=False).apply(compute_ceo_cfo_change)
print("Added 'ceo_cfo_change' dummy variable to merged_df.")

# -------------------------- #
# **5. Final DataFrame Preparation and Saving**
# -------------------------- #

print("Finalizing the merged DataFrame...")
desired_columns = [
    'filtered_topics', 'filtered_texts', 'call_id', 'permco', 'call_date', 'gvkey',
    'fiscal_period_end', 'epsfxq', 'epsfxq_next', 'siccd',
    'similarity_to_overall_average', 'similarity_to_industry_average',
    'similarity_to_company_average', 'prc', 'shrout', 'ret', 'vol',
    'ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term',
    'excess_ret_immediate', 'excess_ret_short_term', 'excess_ret_medium_term', 'excess_ret_long_term',
    'market_cap', 'ceo_participates', 'ceo_cfo_change'  # **Added dummy variables**
]

# Check if all desired columns are present
missing_cols = set(desired_columns) - set(merged_df.columns)
if missing_cols:
    print(f"Warning: The following expected columns are missing in merged_df and will be excluded from the final DataFrame: {missing_cols}")

# Take absolute values for the 'prc' column
if 'prc' in merged_df.columns:
    merged_df['prc'] = merged_df['prc'].abs()

# Select only the columns that exist
final_columns = [col for col in desired_columns if col in merged_df.columns]
merged_df = merged_df[final_columns]

# Example: Log-transform 'market_cap' if desired (Optional)
if 'market_cap' in merged_df.columns:
    merged_df['log_market_cap'] = np.log1p(merged_df['market_cap'])
    print("Added 'log_market_cap' to the DataFrame.")

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
