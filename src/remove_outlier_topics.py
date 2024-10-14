# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:49:19 2024

@author: nikla
"""
# TODO : This script should be put in a function and moved to utils
import pandas as pd 

path = "D:/daten_masterarbeit/topics_output_sentences_50_zeroshot_0_minsim.csv"
output_path = "D:/daten_masterarbeit/topics_output_sentences_50_zeroshot_0_minsim_outlier_removed.csv"
output_path_sample = "D:/daten_masterarbeit/topics_output_sentences_50_zeroshot_0_minsim_outlier_removed_sample.csv"

# Load your CSV file
df = pd.read_csv(path)

# Define the topics to keep
topics_to_keep = [1,2,3,4,5,6]

# Function to keep only the specified topics and corresponding texts
def keep_topics_and_texts(row, topics_to_keep):
    # Convert string representation of list to actual list
    topics = eval(row['topics'])
    texts = eval(row['text'])
    
    # Filter topics and corresponding texts
    filtered_data = [(topic, text) for topic, text in zip(topics, texts) if topic in topics_to_keep]
    
    # Unpack the filtered topics and texts
    if filtered_data:
        filtered_topics, filtered_texts = zip(*filtered_data)
    else:
        filtered_topics, filtered_texts = [], []
    
    return list(filtered_topics), list(filtered_texts)

# Apply the function to each row
df[['filtered_topics', 'filtered_texts']] = df.apply(lambda row: keep_topics_and_texts(row, topics_to_keep), axis=1, result_type='expand')

# Consistency check to validate if topics and texts are of the same length
def check_consistency(row):
    topics = eval(row['topics'])
    texts = eval(row['text'])
    return len(topics) == len(texts)

# Apply the consistency check to each row and create a new column to store the result
df['consistent'] = df.apply(check_consistency, axis=1)

# Display the resulting dataframe with filtered topics, texts, and consistency check
print(df[['topics', 'text', 'filtered_topics', 'filtered_texts', 'consistent']])

#export to csv, pathname should be original path + outlier_removed
df.to_csv(output_path, sep='\t', encoding='utf-8', index=False, header=list(df))
#df.head(100).to_csv(output_path_sample, sep='\t', encoding='utf-8', index=False, header=list(df))