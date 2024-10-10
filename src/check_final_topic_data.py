import pandas as pd
import ast

# Correct file path (use raw string notation to avoid issues with backslashes)
path = r"D:\daten_masterarbeit\topics_output_sentences_50_zeroshot_0_minsim_outlier_removed.csv"

# Load your CSV file
df = pd.read_csv(path, sep='\t', encoding='utf-8')

# Function to safely evaluate and convert string representations of lists into actual lists
def convert_to_list(column):
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Apply the conversion function to the relevant columns
df['filtered_topics'] = convert_to_list(df['filtered_topics'])
df['filtered_texts'] = convert_to_list(df['filtered_texts'])

# Function to print filtered topics and texts for first 10 rows
def print_filtered_output(df):
    for idx, row in df.head(10).iterrows():
        print(f"Row {idx + 1}:")
        filtered_topics = row['filtered_topics']
        filtered_texts = row['filtered_texts']
        
        # Loop through each topic and its corresponding text
        if isinstance(filtered_topics, list) and isinstance(filtered_texts, list):
            for topic, text in zip(filtered_topics, filtered_texts):
                print(f"  Topic: {topic}")
                print(f"  Corresponding Text: {text}")
        print()  # Blank line between rows

# Run the function on your dataframe
print_filtered_output(df)
