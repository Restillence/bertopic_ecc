import pandas as pd
import ast

# File path
file_path = "D:/daten_masterarbeit/topics_output.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Convert 'text' column from string representation of list to actual list
df['text'] = df['text'].apply(ast.literal_eval)

# Print each element of the 'text' column
print("Each element in the 'text' column:")
for idx, text_list in enumerate(df['text']):
    print(f"\nEntry {idx + 1}:")
    for text in text_list:
        print(f" - {text}")

# Basic Statistics
num_elements = df['text'].apply(len).sum()
average_length_per_list = df['text'].apply(len).mean()
average_length_per_element = df['text'].apply(lambda x: sum(len(i) for i in x) / len(x)).mean()
length_distribution_per_list = df['text'].apply(len).describe()
length_distribution_per_element = df['text'].apply(lambda x: [len(i) for i in x]).explode().describe()

# Display results
print(f"\nNumber of total elements in 'text' column: {num_elements}")
print(f"Average number of elements per list: {average_length_per_list:.2f}")
print(f"Average length of elements: {average_length_per_element:.2f} characters")
print("\nLength distribution of lists:")
print(length_distribution_per_list)
print("\nLength distribution of elements:")
print(length_distribution_per_element)

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df['text_length'] = df['text'].apply(len)
plt.hist(df['text_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Text Lengths')
plt.xlabel('Length of Text (characters)')
plt.ylabel('Frequency')
plt.show()
