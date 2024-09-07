"""
This file creates descriptive statistics of the ecc data.
"""

#imports
from file_handling import read_index_file, create_ecc_sample  # Import the file_handling module
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nlp_plots import (
    plot_tfidf_top_terms,
    plot_topics_tsne_pca,
    plot_ner,
    plot_ngram_frequencies,
    plot_sentiment_analysis,
    plot_keyword_cooccurrence,
    plot_word_length_distribution,
    plot_pos_tagging_distribution,
    plot_bag_of_words,
    plot_wordcloud
)


#variables
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 50  # number of unique companies where we want to create our sample from
random_seed = 42  # Set a random seed for reproducibility

#constants
#nothing to change here
index_file_path = index_file_ecc_folder + "list_earnings_call_transcripts.csv"

def load_data(index_file_path, sample_size, folderpath_ecc, random_seed):
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Read the index file
    index_file = read_index_file(index_file_path)
    print("Index file loaded successfully.")

    # Create sample of earnings conference calls
    ecc_sample = create_ecc_sample(sample_size, index_file, folderpath_ecc)
    return ecc_sample

def display_sample(ecc_sample):
    # Display the first 5 Companies of the sample (for demonstration)
    print("\nHere is the sample of earnings conference calls:")
    for i, (permco, calls) in enumerate(ecc_sample.items()):
        if i >= 5:
            break
        for key, value in calls.items():
            print(f"Permco_Key: {key}")
            print(f"Company Name: {value[0]}")
            print(f"Date: {value[1]}")
            print(f"Text Content: {value[2][1000:1100]}...")  # Displaying some letters from the Text.
            print()

def convert_to_dataframe(ecc_sample):
    records = []
    for permco, calls in ecc_sample.items():
        for call_id, values in calls.items():
            company_info, date, text = values
            records.append({
                'permco': permco,
                'call_id': call_id,
                'company_info': company_info,
                'date': date,
                'text': text,
                'text_length': len(text.split())
            })
    results_df = pd.DataFrame(records)
    return results_df

def plot_ecc_length_distribution(results_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['text_length'], bins=30, kde=True)
    plt.title('Distribution of ECC Lengths')
    plt.xlabel('ECC Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_ecc_length_by_company(results_df, top_n=20):
    top_companies = results_df['company_info'].value_counts().nlargest(top_n).index
    top_results_df = results_df[results_df['company_info'].isin(top_companies)]

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='company_info', y='text_length', data=top_results_df)
    plt.title(f'ECC Length Distribution by Top {top_n} Companies')
    plt.xlabel('Company')
    plt.ylabel('ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def plot_ecc_length_over_time(results_df):
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values(by='date')

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='date', y='text_length', data=results_df)
    plt.title('ECC Length Over Time')
    plt.xlabel('Date')
    plt.ylabel('ECC Length (Number of Words)')
    plt.grid(True)
    plt.show()

def plot_files_per_permco(results_df):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='permco', data=results_df, order=results_df['permco'].value_counts().index)
    plt.title('Number of Files per Permco')
    plt.xlabel('Permco')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def plot_average_ecc_length_per_company(results_df):
    avg_length_per_company = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    avg_length_per_company.plot(kind='bar')
    plt.title('Average ECC Length by Top 20 Companies')
    plt.xlabel('Company')
    plt.ylabel('Average ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def plot_ecc_length_distribution_by_year(results_df):
    results_df['year'] = results_df['date'].dt.year
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='year', y='text_length', data=results_df)
    plt.title('ECC Length Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def plot_files_distribution(results_df):
    file_counts = results_df['permco'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.histplot(file_counts, bins=30, kde=True)
    plt.title('Distribution of Files per Permco')
    plt.xlabel('Number of Files')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def additional_descriptive_statistics(results_df):
    num_unique_companies = results_df['company_info'].nunique()

    calls_per_company = results_df['company_info'].value_counts().to_frame().reset_index()
    calls_per_company.columns = ['Company', 'Number of Calls']

    top5_avg_length = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(5).to_frame().reset_index()
    top5_avg_length.columns = ['Company', 'Average ECC Length']

    # Create tables
    summary_stats = results_df.describe().transpose()
    summary_stats_table = pd.DataFrame(summary_stats)

    return num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table

def display_tables(num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table, output_html_path):
    html_content = f"""
    <html>
    <head>
        <title>ECC Descriptive Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            h1 {{ text-align: center; }}
        </style>
    </head>
    <body>
        <h1>Summary Statistics</h1>
        {summary_stats_table.to_html(classes='table')}
        <h1>Number of Calls per Company</h1>
        {calls_per_company.to_html(index=False, classes='table')}
        <h1>Top 5 Companies by Average ECC Length</h1>
        {top5_avg_length.to_html(index=False, classes='table')}
        <h1>Number of Unique Companies</h1>
        <p>{num_unique_companies}</p>
    </body>
    </html>
    """

    with open(output_html_path, "w") as file:
        file.write(html_content)

    print("\nSummary Statistics:")
    print(summary_stats_table)
    print(f"\nNumber of unique companies: {num_unique_companies}")
    print("\nNumber of calls per company:")
    print(calls_per_company.head(10).to_string(index=False))  # Displaying top 10 for brevity
    print("\nTop 5 companies by average ECC length:")
    print(top5_avg_length.to_string(index=False))
    print(f"\nTables have been saved to {output_html_path}")

def main():
    ecc_sample = load_data(index_file_path, sample_size, folderpath_ecc, random_seed)
    display_sample(ecc_sample)
    results_df = convert_to_dataframe(ecc_sample)
    plot_ecc_length_distribution(results_df)
    plot_ecc_length_by_company(results_df)
    plot_ecc_length_over_time(results_df)
    plot_files_distribution(results_df)
    plot_average_ecc_length_per_company(results_df)
    plot_ecc_length_distribution_by_year(results_df)
    plot_bag_of_words(results_df) #add it back later
    plot_wordcloud(results_df) #add it back later
    plot_tfidf_top_terms(results_df)
    plot_topics_tsne_pca(results_df)
    #plot_ner(results_df) #very computational intensive!
    plot_ngram_frequencies(results_df, n=2)
    plot_ngram_frequencies(results_df, n=3)
    plot_sentiment_analysis(results_df) #add it back later
    plot_keyword_cooccurrence(results_df)
    plot_word_length_distribution(results_df)
    #plot_pos_tagging_distribution(results_df) #very computational intensive!
    
    num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table = additional_descriptive_statistics(results_df)
    output_html_path = os.path.join(index_file_ecc_folder, 'ecc_statistics.html')
    display_tables(num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table, output_html_path)
    print(f"HTML report has been saved to {output_html_path}")

if __name__ == "__main__":
    main()
