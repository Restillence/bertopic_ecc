# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:49:19 2024

@author: nikla
"""
#TODO : This script should be put in a function and moved to utils
import pandas as pd 

path = "D:/daten_masterarbeit/topics_output.csv"

# Load your CSV file
df = pd.read_csv(path)

# Define the topics to remove
topics_to_remove = [16,146,1, 5, -2, 4]

# Function to remove the topics and corresponding texts
def remove_topics_and_texts(row, topics_to_remove):
    # Convert string representation of list to actual list
    topics = eval(row['topics'])
    texts = eval(row['text'])
    
    # Filter topics and corresponding texts
    filtered_data = [(topic, text) for topic, text in zip(topics, texts) if topic not in topics_to_remove]
    
    # Unpack the filtered topics and texts
    if filtered_data:
        filtered_topics, filtered_texts = zip(*filtered_data)
    else:
        filtered_topics, filtered_texts = [], []
    
    return list(filtered_topics), list(filtered_texts)

# Apply the function to each row
df[['filtered_topics', 'filtered_texts']] = df.apply(lambda row: remove_topics_and_texts(row, topics_to_remove), axis=1, result_type='expand')

# Consistency check to validate if topics and texts are of the same length
def check_consistency(row):
    topics = eval(row['topics'])
    texts = eval(row['text'])
    return len(topics) == len(texts)

# Apply the consistency check to each row and create a new column to store the result
df['consistent'] = df.apply(check_consistency, axis=1)

# Display the resulting dataframe with filtered topics, texts, and consistency check
print(df[['topics', 'text', 'filtered_topics', 'filtered_texts', 'consistent']])

