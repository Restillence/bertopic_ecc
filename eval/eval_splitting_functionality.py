import pandas as pd
import ast  # to safely evaluate string representations of lists

# File path
file_path = "D:/daten_masterarbeit/topics_output.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Check the first few rows to ensure the 'text' column is read correctly
print("First few rows of the DataFrame:")
print(df.head())

# Convert the 'text' column from string to list (if it's indeed a string representation of a list)
df["text"] = df["text"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert to list of lists
text_list = df["text"].tolist()

for filing in text_list:
    print("##################################NEW FILING####################################")
    print(filing[0][:500])  # Print the first 500 characters of the first element


try:
    with open("evalfile_splitting_functionality.txt", "w", encoding="utf-8") as f:
        for filing in text_list:
            f.write("##################################NEW FILING####################################\n")
            for element in filing:
                f.write(element + "\n\n")
    print("File created successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
