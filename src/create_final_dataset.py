# create_final_dataset.py

import json
import pandas as pd
import numpy as np
import sys
import warnings
import ast
from utils import (
    process_topics, compute_similarity_to_average, count_word_length_text, 
    count_items, map_topics_to_clusters, remove_neg_one_from_columns
)

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

# Read 'modeling_type' from config
model_type = config.get('modeling_type', 'zeroshot').lower()

# Validate model_type
if model_type not in ['zeroshot', 'regular']:
    print(f"Invalid 'modeling_type' in config: {model_type}. Defaulting to 'zeroshot'.")
    model_type = 'zeroshot'
else:
    print(f"Model type selected: {model_type}")

# Process the topics
print("Processing topics...")
try:
    processed_df = process_topics(topic_input_path, topic_output_path, topics_to_keep, topic_threshold_percentage)
    print(f"Processed DataFrame columns: {processed_df.columns}")
except KeyError as e:
    print(f"Error during processing topics: {e}")
    sys.exit(1)

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

# Check for missing 'gvkey's
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

# Check uniqueness of 'call_id' in processed_df
unique_call_ids_processed = processed_df['call_id'].nunique()
total_call_ids_processed = len(processed_df)
print(f"Processed_df: {unique_call_ids_processed} unique 'call_id's out of {total_call_ids_processed} total rows.")

# Remove duplicates in processed_df based on 'call_id' to ensure uniqueness
processed_df = processed_df.drop_duplicates(subset=['call_id'])
print(f"Processed_df after dropping duplicates based on 'call_id': {len(processed_df)} rows.")

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

# Select necessary columns from processed_df, including 'presentation_text' and additional columns
processed_df_columns = [
    'call_id', 'gvkey', 'call_date', 'call_date_with_time', 
    'permco', 'filtered_presentation_topics', 'filtered_texts', 'presentation_text', 
    'ceo_participates', 'ceo_names', 'cfo_names',
    'participant_question_topics', 'management_answer_topics'
]
processed_df = processed_df[processed_df_columns]
print(f"Selected columns from processed_df: {processed_df_columns}")

# Ensure 'participant_question_topics' and 'management_answer_topics' are lists
print("Ensuring 'participant_question_topics' and 'management_answer_topics' are lists...")
processed_df['participant_question_topics'] = processed_df['participant_question_topics'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
processed_df['management_answer_topics'] = processed_df['management_answer_topics'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Replace NaNs with empty lists
processed_df['participant_question_topics'] = processed_df['participant_question_topics'].apply(
    lambda x: x if isinstance(x, list) else []
)
processed_df['management_answer_topics'] = processed_df['management_answer_topics'].apply(
    lambda x: x if isinstance(x, list) else []
)

# Compute 'word_length_presentation' by counting words in 'presentation_text'
print("Computing 'word_length_presentation' by counting words in 'presentation_text'...")
processed_df['word_length_presentation'] = processed_df['presentation_text'].apply(count_word_length_text)
print("Added 'word_length_presentation' column to processed_df.")

# Compute counts for participant questions and management answers
print("Computing 'length_participant_questions' by counting items in 'participant_question_topics'...")
processed_df['length_participant_questions'] = processed_df['participant_question_topics'].apply(count_items)
print("Added 'length_participant_questions' column to processed_df.")

print("Computing 'length_management_answers' by counting items in 'management_answer_topics'...")
processed_df['length_management_answers'] = processed_df['management_answer_topics'].apply(count_items)
print("Added 'length_management_answers' column to processed_df.")

# Drop 'presentation_text' column as it's no longer needed
print("Dropping the 'presentation_text' column from processed_df...")
processed_df = processed_df.drop(columns=['presentation_text'], errors='ignore')
print("Dropped 'presentation_text' column.")

# Select necessary columns from df_crsp_monthly
df_crsp_monthly_columns = ['gvkey', 'datadate', 'epsfxq', 'epsfxq_next', 'siccd', 'permco']
df_crsp_monthly = df_crsp_monthly[df_crsp_monthly_columns]

# Merge processed_df with df_crsp_monthly
print("Performing custom merge between processed_df and df_crsp_monthly...")
# Step 1: Perform a full merge on 'gvkey' to create all possible combinations.
temp_merged = pd.merge(
    processed_df.drop(columns=['call_date_with_time']),  # Exclude 'call_date_with_time' to avoid duplication
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
        lambda row: siccd_mapping.get(row['permco'], np.nan) if pd.isna(row['siccd']) else row['siccd'],
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

apply_mapping = False
if topics_to_keep == "auto":
    apply_mapping = True

# Apply topic mapping conditionally
print("Applying topic mapping if required...")
merged_df = map_topics_to_clusters(merged_df, model=model_type, apply_mapping=apply_mapping)

# Ensure 'filtered_presentation_topics' is evaluated as lists
merged_df['filtered_presentation_topics'] = merged_df['filtered_presentation_topics'].apply(
    lambda x: x if isinstance(x, list) else []
)

# Determine the number of topics after mapping
try:
    unique_topics = set()
    merged_df['filtered_presentation_topics'].apply(lambda x: unique_topics.update(x))
    num_topics = max(unique_topics) + 1
except Exception as e:
    print(f"Error determining number of topics: {e}")
    num_topics = 0  # Fallback or handle accordingly
print(f"Number of topics determined after mapping: {num_topics}")

# Convert 'siccd' to integer (if possible)
merged_df['siccd'] = merged_df['siccd'].astype(float).astype('Int64')

# Compute similarities
print("Computing similarities to overall, industry and firm-specific averages...")
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
# Assuming 'ret' is the return column in df_crsp_daily
if 'ret' in df_crsp_daily.columns:
    df_crsp_daily['ret'] = pd.to_numeric(df_crsp_daily['ret'], errors='coerce')
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
else:
    print("Warning: 'ret' column not found in df_crsp_daily. Skipping market returns computation.")

# Compute rolling beta
print("Computing rolling beta for each stock...")

# Ensure 'ret' and 'market_ret' are numeric
df_crsp_daily['ret'] = pd.to_numeric(df_crsp_daily['ret'], errors='coerce')
df_crsp_daily['market_ret'] = pd.to_numeric(df_crsp_daily['market_ret'], errors='coerce')

# Sort the DataFrame by 'permco' and 'date'
df_crsp_daily = df_crsp_daily.sort_values(['permco', 'date']).reset_index(drop=True)

def compute_rolling_beta(group):
    # group is the data for one 'permco'
    # Set the window size, e.g., 252 days
    window = 252
    # Compute rolling covariance and variance
    rolling_cov = group['ret'].rolling(window).cov(group['market_ret'])
    rolling_var = group['market_ret'].rolling(window).var()
    group['rolling_beta'] = rolling_cov / rolling_var
    return group

df_crsp_daily = df_crsp_daily.groupby('permco', group_keys=False).apply(compute_rolling_beta)
print("Computed rolling beta for each stock.")

# Compute future returns
print("Computing future returns...")
df_crsp_daily = df_crsp_daily.sort_values(['permco', 'date']).reset_index(drop=True)

def compute_future_returns(group):
    group = group.sort_values('date').reset_index(drop=True)
    n = len(group)
    ret_values = group['ret'].values
    market_ret_values = group['market_ret'].values

    ret_immediate = np.full(n, np.nan)
    ret_short_term = np.full(n, np.nan)
    ret_medium_term = np.full(n, np.nan)
    ret_long_term = np.full(n, np.nan)

    excess_ret_immediate = np.full(n, np.nan)
    excess_ret_short_term = np.full(n, np.nan)
    excess_ret_medium_term = np.full(n, np.nan)
    excess_ret_long_term = np.full(n, np.nan)

    for i in range(n):
        # Immediate market reaction: t-1 to t+1
        if i - 1 >= 0 and i + 1 < n:
            returns = ret_values[i - 1 : i + 2]
            market_returns = market_ret_values[i - 1 : i + 2]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_immediate[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_immediate[i] = ret_immediate[i] - market_return
        
        # Short-term market reaction: t+2 to t+6
        if i + 6 < n:
            returns = ret_values[i + 2 : i + 7]
            market_returns = market_ret_values[i + 2 : i + 7]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_short_term[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_short_term[i] = ret_short_term[i] - market_return
        
        # Medium-term market reaction: t+2 to t+21
        if i + 21 < n:
            returns = ret_values[i + 2 : i + 22]
            market_returns = market_ret_values[i + 2 : i + 22]
            if not np.isnan(returns).any() and not np.isnan(market_returns).any():
                ret_medium_term[i] = np.prod(1 + returns) - 1
                market_return = np.prod(1 + market_returns) - 1
                excess_ret_medium_term[i] = ret_medium_term[i] - market_return
        
        # Long-term market reaction: t+2 to t+60
        if i + 61 < n:
            returns = ret_values[i + 2 : i + 62]
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

# Merge future returns and rolling beta into merged_df
print("Merging future returns into the merged DataFrame...")
merged_df = pd.merge(
    merged_df,
    df_crsp_daily[[
        'permco', 'date', 'shrout', 'prc', 'vol', 'market_cap', 'ret',
        'ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term', 'market_ret',
        'excess_ret_immediate', 'excess_ret_short_term', 'excess_ret_medium_term', 'excess_ret_long_term',
        'rolling_beta'  # Include rolling_beta
    ]],
    left_on=['permco', 'call_date'],
    right_on=['permco', 'date'],
    how='left'
)
merged_df = merged_df.drop(columns=['date'], errors='ignore')
print(f"Number of rows after merging future returns: {len(merged_df)}")

# Restore 'call_date' with time component
print("Restoring 'call_date' with time component from 'processed_df'...")

# Drop 'call_date_with_time' from merged_df if exists
merged_df = merged_df.drop(columns=['call_date_with_time'], errors='ignore')

# Merge 'call_date_with_time' back into merged_df using 'call_id' and include additional columns
merged_df = pd.merge(
    merged_df,
    processed_df[['call_id', 'call_date_with_time', 'ceo_participates', 'ceo_names', 'cfo_names']],  # Include additional columns
    on='call_id',
    how='left',
    validate='many_to_one'  # Ensure 'call_id's in merged_df are unique
)

# Replace 'call_date' in merged_df with 'call_date_with_time'
merged_df['call_date'] = merged_df['call_date_with_time']

# Drop 'call_date_with_time' column
merged_df = merged_df.drop(columns=['call_date_with_time'], errors='ignore')
print("Restored 'call_date' with time component in merged_df.")

# Consolidate 'ceo_names' and 'cfo_names'
merged_df['ceo_names'] = merged_df['ceo_names_y'].fillna(merged_df['ceo_names_x'])
merged_df['cfo_names'] = merged_df['cfo_names_y'].fillna(merged_df['cfo_names_x'])

# Consolidate 'ceo_participates'
merged_df['ceo_participates'] = merged_df['ceo_participates_y'].fillna(merged_df['ceo_participates_x'])

# Drop the duplicated columns
merged_df = merged_df.drop(columns=['ceo_names_x', 'ceo_names_y', 'cfo_names_x', 'cfo_names_y',
                                    'ceo_participates_x', 'ceo_participates_y'], errors='ignore')
print("Consolidated 'ceo_names', 'cfo_names', and 'ceo_participates' into single columns and dropped duplicates.")

# Add CEO/CFO Change Dummy Variables
# Ensure merged_df is sorted by 'gvkey' and 'call_date'
merged_df = merged_df.sort_values(by=['gvkey', 'call_date']).reset_index(drop=True)
print("DataFrame sorted by 'gvkey' and 'call_date'.")

# Create shifted columns for previous CEO and CFO names
merged_df['prev_ceo_names'] = merged_df.groupby('gvkey')['ceo_names'].shift(1)
merged_df['prev_cfo_names'] = merged_df.groupby('gvkey')['cfo_names'].shift(1)

# Define function to compare lists
def lists_are_different(list1, list2):
    # Handle cases where one or both lists are empty or NaN
    if not isinstance(list1, list):
        list1 = []
    if not isinstance(list2, list):
        list2 = []
    return list1 != list2

# Compute 'ceo_change'
merged_df['ceo_change'] = merged_df.apply(
    lambda row: 1 if lists_are_different(row['ceo_names'], row['prev_ceo_names']) else 0, axis=1
)

# Compute 'cfo_change'
merged_df['cfo_change'] = merged_df.apply(
    lambda row: 1 if lists_are_different(row['cfo_names'], row['prev_cfo_names']) else 0, axis=1
)

# Fill NaN for the first call per 'gvkey' with 0
merged_df['ceo_change'] = merged_df['ceo_change'].fillna(0).astype(int)
merged_df['cfo_change'] = merged_df['cfo_change'].fillna(0).astype(int)

# Create 'ceo_cfo_change' as 1 if either CEO or CFO changed
merged_df['ceo_cfo_change'] = (merged_df['ceo_change'] | merged_df['cfo_change']).astype(int)

# Drop the helper columns
merged_df = merged_df.drop(columns=['prev_ceo_names', 'prev_cfo_names'], errors='ignore')
print("Added 'ceo_change', 'cfo_change', and 'ceo_cfo_change' dummy variables.")

# Remove -1 from 'participant_question_topics' and 'management_answer_topics'
print("Removing -1 from 'participant_question_topics' and 'management_answer_topics'...")
merged_df = remove_neg_one_from_columns(
    merged_df, 
    ['participant_question_topics', 'management_answer_topics']
)
print("Removed -1 from specified topic columns.")

# Take absolute values for the 'prc' column
if 'prc' in merged_df.columns:
    merged_df['prc'] = merged_df['prc'].abs()

# Final DataFrame Preparation and Saving
print("Finalizing the merged DataFrame...")
desired_columns = [
    'filtered_presentation_topics', 'filtered_texts',
    'call_id', 'permco', 'call_date', 'gvkey',
    'ceo_participates', 'ceo_names', 'cfo_names',
    'fiscal_period_end', 'epsfxq', 'epsfxq_next', 'siccd',
    'similarity_to_overall_average', 'similarity_to_industry_average',
    'similarity_to_company_average', 'prc', 'shrout', 'ret', 'vol', 'market_cap',
    'ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term', 'market_ret',
    'excess_ret_immediate', 'excess_ret_short_term', 'excess_ret_medium_term', 'excess_ret_long_term',
    'rolling_beta',
    'word_length_presentation',
    'length_participant_questions', 'length_management_answers',
    'ceo_change', 'cfo_change', 'ceo_cfo_change',
    'participant_question_topics', 'management_answer_topics'
]

# Select only the columns that exist
final_columns = [col for col in desired_columns if col in merged_df.columns]
merged_df = merged_df[final_columns]

# Remove the first year (2022) of the sample before saving
print("Removing data from 2022 from merged_df before saving...")
merged_df['call_date'] = pd.to_datetime(merged_df['call_date'])
merged_df = merged_df[merged_df['call_date'].dt.year != 2022]
print(f"Number of rows after removing 2022 data: {len(merged_df)}")

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
