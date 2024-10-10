# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:29:41 2024

@author: nikla
"""

#finance data integration exploration

import pandas as pd

# File paths
file_path_topics = "D:/daten_masterarbeit/topics_output.csv"
file_path_crsp_daily = "D:/daten_masterarbeit/CRSP_daily_master_thesis.csv"
merged_file_path = "D:/daten_masterarbeit/merged_topics_crsp.csv"

# Read the CSV files
df_topics = pd.read_csv(file_path_topics)
print("Topics file loaded successfully.")
df_crsp_daily = pd.read_csv(file_path_crsp_daily)
print("CRSP/Monthly file loaded successfully.")

# Ensure permco columns are of the same data type
df_topics['permco'] = df_topics['permco'].astype(str)
df_crsp_daily['permco'] = df_crsp_daily['permco'].astype(str)

# Select relevant columns from CRSP/Compustat data
print("Selecting relevant columns from CRSP/Compustat data...")
cols_to_keep = ['date', 'siccd', 'ncusip', 'permco', 'prc', 'vol', 'ret', 'cfacpr', 'cfacshr', 'gvkey', 'month_id_datadate', 'datadate', 'epsfxq', 'niq']
df_crsp_monthly = df_crsp_daily[cols_to_keep]

# If there are duplicate column names after merging, such as multiple datadate columns, rename them or drop duplicates
df_crsp_monthly = df_crsp_monthly.rename(columns={'datadate': 'datadate_crsp'})

# Merge the DataFrames on permco
print("Merging the DataFrames on permco...")
merged_df = pd.merge(df_topics, df_crsp_monthly, on='permco', how='inner')

# Inspect the merged DataFrame
print(merged_df.head())

# Save the merged DataFrame if needed
print("Saving the merged DataFrame...")
merged_df.to_csv(merged_file_path, index=False)
