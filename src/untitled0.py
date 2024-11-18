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

# Print all column names
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

# Columns that should be lists
list_columns = ['analyst_questions', 'question_topics', 'presentation_text', 'presentation_topics']

# Convert the string representations to actual lists
for col in list_columns:
    df[col] = df[col].apply(convert_to_list)

# Verify conversion
print(type(df['analyst_questions'].iloc[0]))  # Should print <class 'list'>
print(type(df['question_topics'].iloc[0]))    # Should print <class 'list'>

# Compare lengths for 'analyst_questions' and 'question_topics'
print("Comparing 'analyst_questions' and 'question_topics':")
for i in range(len(df)):
    aq_length = len(df['analyst_questions'][i])
    qt_length = len(df['question_topics'][i])
    
    if aq_length == qt_length:
        print(f"Row {i}: same length ({aq_length})")
    else:
        print(f"Row {i}: different length (analyst_questions: {aq_length}, question_topics: {qt_length})")

# Compare lengths for 'presentation_text' and 'presentation_topics'
print("\nComparing 'presentation_text' and 'presentation_topics':")
for i in range(len(df)):
    pt_length = len(df['presentation_text'][i])
    ppt_length = len(df['presentation_topics'][i])
    
    if pt_length == ppt_length:
        print(f"Row {i}: same length ({pt_length})")
    else:
        print(f"Row {i}: different length (presentation_text: {pt_length}, presentation_topics: {ppt_length})")
