# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:59:33 2024

@author: nikla
"""

import pandas as pd
import ast
import json

# Define the file path
file_path = "D:/daten_masterarbeit/topics_output.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Print all column names for verification
print("Column Names in the CSV:")
print(df.columns)

# Function to convert string to list
def convert_to_list(cell):
    if pd.isna(cell):
        return []  # Handle NaN or missing values
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            return [cell]  # Fallback: treat as a single-element list

# **Updated** Columns that should be lists
list_columns = ['participant_questions', 'participant_question_topics', 'presentation_text', 'presentation_topics']

# Check if the updated columns exist in the DataFrame
missing_columns = [col for col in list_columns if col not in df.columns]
if missing_columns:
    print(f"Error: The following expected columns are missing in the CSV: {missing_columns}")
    # Optionally, exit the script or handle accordingly
    import sys
    sys.exit(1)

# Convert the string representations to actual lists
for col in list_columns:
    df[col] = df[col].apply(convert_to_list)

# Verify conversion
print("\nType Verification:")
print(f"Type of 'participant_questions' in first row: {type(df['participant_questions'].iloc[0])}")
print(f"Type of 'participant_question_topics' in first row: {type(df['participant_question_topics'].iloc[0])}")

# Compare lengths for 'participant_questions' and 'participant_question_topics'
print("\nComparing 'participant_questions' and 'participant_question_topics':")
for i in range(len(df)):
    pq_length = len(df['participant_questions'][i])
    pqt_length = len(df['participant_question_topics'][i])
    
    if pq_length == pqt_length:
        print(f"Row {i}: same length ({pq_length})")
    else:
        print(f"Row {i}: different length (participant_questions: {pq_length}, participant_question_topics: {pqt_length})")

# Compare lengths for 'presentation_text' and 'presentation_topics'
print("\nComparing 'presentation_text' and 'presentation_topics':")
for i in range(len(df)):
    pt_length = len(df['presentation_text'][i])
    ppt_length = len(df['presentation_topics'][i])
    
    if pt_length == ppt_length:
        print(f"Row {i}: same length ({pt_length})")
    else:
        print(f"Row {i}: different length (presentation_text: {pt_length}, presentation_topics: {ppt_length})")

"""
#print 1 element of participant_questions and participant_question_topics
print("\nFirst element of 'participant_questions' and 'participant_question_topics':")
print(df['participant_questions'].iloc[0][0])
print(df['participant_question_topics'].iloc[0][0])
"""
#print call_id
df["call_id"][0]
"""
for i in range(20):
    for j in range(20):
        print(df['participant_questions'].iloc[i][j])
        print(df['participant_question_topics'].iloc[i][j])
    
"""