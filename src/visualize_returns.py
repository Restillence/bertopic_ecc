# visualize_returns.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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
merged_file_path = config['merged_file_path']

# Read the final dataset
print("Reading the final merged DataFrame...")
df = pd.read_csv(merged_file_path)
print(f"DataFrame loaded with {len(df)} rows.")

# Convert 'call_date' to datetime if necessary
if df['call_date'].dtype == 'object':
    df['call_date'] = pd.to_datetime(df['call_date'], errors='coerce')

# Define similarity measures
similarity_measures = ['similarity_to_overall_average', 'similarity_to_industry_average', 'similarity_to_company_average']

# Define return periods with updated variable names
return_periods = ['ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term']

# Initialize a dictionary to hold descriptive statistics
descriptive_stats = {}

# Function to classify high and low similarity groups
def classify_similarity(df, similarity_col):
    mean = df[similarity_col].mean()
    std = df[similarity_col].std()
    high_threshold = mean + 1 * std
    low_threshold = mean - 1 * std

    df['similarity_group'] = 'Medium'
    df.loc[df[similarity_col] >= high_threshold, 'similarity_group'] = 'High Similarity'
    df.loc[df[similarity_col] <= low_threshold, 'similarity_group'] = 'Low Similarity'

    # Filter out 'Medium' group if needed
    df_filtered = df[df['similarity_group'] != 'Medium']

    return df_filtered

# Loop through each similarity measure
for similarity_col in similarity_measures:
    print(f"\nProcessing similarity measure: {similarity_col}")

    # Check if the similarity column exists
    if similarity_col not in df.columns:
        print(f"Column '{similarity_col}' not found in DataFrame. Skipping.")
        continue

    # Classify stocks into high and low similarity groups
    df_similarity = classify_similarity(df.copy(), similarity_col)

    # Check if we have enough data
    if df_similarity.empty:
        print(f"No data available after filtering for similarity measure '{similarity_col}'.")
        continue

    # Calculate descriptive statistics
    stats = df_similarity.groupby('similarity_group')[similarity_col].describe()
    descriptive_stats[similarity_col] = stats
    print(f"Descriptive statistics for {similarity_col}:\n{stats}")

    # Plotting
    plt.figure(figsize=(12, 6))

    for similarity_group in ['High Similarity', 'Low Similarity']:
        group_df = df_similarity[df_similarity['similarity_group'] == similarity_group]

        # Compute average cumulative returns for each return period
        avg_returns = []
        periods = []

        for ret_col in return_periods:
            if ret_col in group_df.columns:
                avg_return = group_df[ret_col].mean()
                avg_returns.append(avg_return)
                periods.append(ret_col)
            else:
                print(f"Return column '{ret_col}' not found in DataFrame.")

        if avg_returns:
            # Plot the average cumulative returns
            plt.plot(periods, avg_returns, marker='o', label=similarity_group)
        else:
            print(f"No return data available for similarity group '{similarity_group}' and similarity measure '{similarity_col}'.")

    plt.title(f'Average Cumulative Returns by Similarity Group\n(Similarity Measure: {similarity_col})')
    plt.xlabel('Return Period')
    plt.ylabel('Average Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Optionally, save descriptive statistics to a CSV file
print("\nSaving descriptive statistics to 'descriptive_statistics.csv'...")
desc_stats_df = pd.concat(descriptive_stats)
desc_stats_df.to_csv('descriptive_statistics.csv')
print("Descriptive statistics saved.")

print("Visualization script completed successfully.")
