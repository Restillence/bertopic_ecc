import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Define the file path
file_path = r'D:\daten_masterarbeit\final_dataset.csv'  # Use raw string to handle backslashes

# Load the dataset
df_final = pd.read_csv(file_path)

# Convert list-like columns from string to list if necessary
list_cols = ['filtered_presentation_topics', 'filtered_texts', 'ceo_names', 'cfo_names',
             'participant_question_topics', 'management_answer_topics']

for col in list_cols:
    df_final[col] = df_final[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert date columns to datetime
df_final['call_date'] = pd.to_datetime(df_final['call_date'])
df_final['fiscal_period_end'] = pd.to_datetime(df_final['fiscal_period_end'])

# ------------------ Descriptive Statistics ------------------

# Summary Statistics
numerical_cols = [
    'epsfxq', 'epsfxq_next', 'prc', 'shrout', 'ret', 'vol', 'market_cap',
    'ret_immediate', 'ret_short_term', 'ret_medium_term', 'ret_long_term',
    'market_ret', 'excess_ret_immediate', 'excess_ret_short_term',
    'excess_ret_medium_term', 'excess_ret_long_term', 'word_length_presentation',
    'length_participant_questions', 'length_management_answers',
    'similarity_to_overall_average', 'similarity_to_industry_average',
    'similarity_to_company_average'
]

summary_stats = df_final[numerical_cols].describe().T
print(summary_stats)

# Frequency Counts
categorical_cols = ['ceo_participates', 'ceo_change', 'cfo_change', 'ceo_cfo_change', 'siccd']

for col in categorical_cols:
    print(f"\nFrequency counts for '{col}':")
    print(df_final[col].value_counts(dropna=False))

# Date Range
print(f"\nCall Date Range: {df_final['call_date'].min()} to {df_final['call_date'].max()}")
print(f"Fiscal Period End Range: {df_final['fiscal_period_end'].min()} to {df_final['fiscal_period_end'].max()}")

# Calls Per Year
calls_per_year = df_final['call_date'].dt.year.value_counts().sort_index()
print("\nNumber of Calls Per Year:")
print(calls_per_year)

# ------------------ Visualizations ------------------

sns.set(style="whitegrid")

# Histograms
for col in ['ret', 'market_ret', 'excess_ret_immediate', 'excess_ret_short_term',
            'excess_ret_medium_term', 'excess_ret_long_term',
            'word_length_presentation', 'length_participant_questions', 'length_management_answers']:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_final[col].dropna(), kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Boxplots
for col in ['ret', 'market_ret', 'excess_ret_immediate', 'excess_ret_short_term',
            'excess_ret_medium_term', 'excess_ret_long_term',
            'word_length_presentation', 'length_participant_questions', 'length_management_answers']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_final[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Correlation Heatmap
corr_matrix = df_final[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# Scatter Plot: ret vs market_ret
plt.figure(figsize=(8, 6))
sns.scatterplot(x='market_ret', y='ret', data=df_final, alpha=0.5)
plt.title('Firm Return vs Market Return')
plt.xlabel('Market Return')
plt.ylabel('Firm Return')
plt.show()

# Pair Plot
sns.pairplot(df_final[['ret', 'market_ret', 'excess_ret_immediate', 'excess_ret_short_term',
                       'excess_ret_medium_term', 'excess_ret_long_term']].dropna())
plt.suptitle('Pair Plot of Selected Numerical Variables', y=1.02)
plt.show()

# Time Series: Average Firm Returns Over Time
plt.figure(figsize=(12, 6))
df_final.set_index('call_date')['ret'].resample('M').mean().plot()
plt.title('Average Firm Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Average Return')
plt.tight_layout()
plt.show()

# Time Series: Average Market Returns Over Time
plt.figure(figsize=(12, 6))
df_final.set_index('call_date')['market_ret'].resample('M').mean().plot(color='orange')
plt.title('Average Market Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Average Market Return')
plt.tight_layout()
plt.show()

# Bar Plot: CEO Participates
plt.figure(figsize=(6, 4))
sns.countplot(x='ceo_participates', data=df_final)
plt.title('CEO Participates in Call')
plt.xlabel('CEO Participates (0=No, 1=Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Bar Plot: CEO Change
plt.figure(figsize=(6, 4))
sns.countplot(x='ceo_change', data=df_final)
plt.title('CEO Change Occurred')
plt.xlabel('CEO Change (0=No, 1=Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Bar Plot: Top 10 Industries by SIC Code
top_siccd = df_final['siccd'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_siccd.index.astype(str), y=top_siccd.values, palette='viridis')
plt.title('Top 10 Industries by SIC Code')
plt.xlabel('SIC Code')
plt.ylabel('Number of Calls')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Similarity Measures Distributions
similarity_cols = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]

for col in similarity_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_final[col].dropna(), kde=True, bins=30, color='green')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# CEO/CFO Change Impact on Excess Immediate Return
def categorize_change(change):
    return 'Change' if change == 1 else 'No Change'

df_final['ceo_change_cat'] = df_final['ceo_change'].apply(categorize_change)
df_final['cfo_change_cat'] = df_final['cfo_change'].apply(categorize_change)

# CEO Change vs Excess Immediate Return
plt.figure(figsize=(8, 6))
sns.boxplot(x='ceo_change_cat', y='excess_ret_immediate', data=df_final)
plt.title('Excess Immediate Return by CEO Change')
plt.xlabel('CEO Change')
plt.ylabel('Excess Immediate Return')
plt.tight_layout()
plt.show()

# CFO Change vs Excess Immediate Return
plt.figure(figsize=(8, 6))
sns.boxplot(x='cfo_change_cat', y='excess_ret_immediate', data=df_final)
plt.title('Excess Immediate Return by CFO Change')
plt.xlabel('CFO Change')
plt.ylabel('Excess Immediate Return')
plt.tight_layout()
plt.show()

# Word Length Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df_final['word_length_presentation'].dropna(), kde=True, bins=30, color='purple')
plt.title('Distribution of Word Length in Presentations')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_final['word_length_presentation'])
plt.title('Boxplot of Word Length in Presentations')
plt.xlabel('Word Length')
plt.tight_layout()
plt.show()

# Participant Questions and Management Answers Counts
plt.figure(figsize=(8, 4))
sns.histplot(df_final['length_participant_questions'].dropna(), kde=True, bins=30, color='teal')
plt.title('Distribution of Number of Participant Questions')
plt.xlabel('Number of Questions')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df_final['length_management_answers'].dropna(), kde=True, bins=30, color='coral')
plt.title('Distribution of Number of Management Answers')
plt.xlabel('Number of Answers')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Topic Analysis: Top 10 Most Frequent Presentation Topics
all_topics = [topic for sublist in df_final['filtered_presentation_topics'] for topic in sublist if isinstance(topic, int)]
topic_counts = Counter(all_topics)
most_common_topics = topic_counts.most_common(10)
topic_df = pd.DataFrame(most_common_topics, columns=['Topic', 'Count'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Topic', y='Count', data=topic_df, palette='deep')
plt.title('Top 10 Most Frequent Presentation Topics')
plt.xlabel('Topic ID')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# CEO/CFO Participation Impact on Excess Immediate Return
plt.figure(figsize=(8, 6))
sns.boxplot(x='ceo_participates', y='excess_ret_immediate', data=df_final)
plt.title('Excess Immediate Return by CEO Participation')
plt.xlabel('CEO Participates (0=No, 1=Yes)')
plt.ylabel('Excess Immediate Return')
plt.tight_layout()
plt.show()
