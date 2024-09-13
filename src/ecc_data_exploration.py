#ecc_data_exploration.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from nlp_plots import (
    plot_tfidf_top_terms,
    plot_topics_tsne_pca,
    plot_ngram_frequencies,
    plot_sentiment_analysis,
    plot_keyword_cooccurrence,
    plot_word_length_distribution,
    plot_bag_of_words,
    plot_wordcloud,
    plot_pos_tagging_distribution,
    plot_ner
)
from file_handling import FileHandler
from text_processing import TextProcessor

# Load the config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Extract variables from config.json
folderpath_ecc = config["folderpath_ecc"]
index_file_ecc_folder = config["index_file_ecc_folder"]
ecc_plots_folder = config["ecc_plots_folder"]  # Add plot folder path
sample_size = config["sample_size"]
random_seed = config["random_seed"]
index_file_path = os.path.join(index_file_ecc_folder, config["index_file_path"])

# Assuming TextProcessor is initialized with method and section from config.json
text_processor = TextProcessor(method=config["document_split"], section_to_analyze=config["section_to_analyze"])

# Assuming FileHandler is initialized with the paths from config.json
file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)

def load_data(file_handler, sample_size, random_seed):
    np.random.seed(random_seed)
    ecc_sample = file_handler.create_ecc_sample(sample_size)
    return ecc_sample

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
    return pd.DataFrame(records)

def plot_paragraph_length_distribution(text_processor, results_df):
    print("Computing paragraph word length distribution...")
    paragraph_lengths = []

    for _, row in results_df.iterrows():
        paragraphs = text_processor.split_text(row['text'])  
        paragraph_lengths.extend([len(paragraph.split()) for paragraph in paragraphs])

    plt.figure(figsize=(10, 6))
    sns.histplot(paragraph_lengths, bins=30, kde=True)
    plt.title('Distribution of Paragraph Word Lengths')
    plt.xlabel('Paragraph Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'paragraph_word_length_distribution.png')
    plt.savefig(plot_path)
    plt.close()

def plot_ecc_length_distribution(results_df):
    print("Computing ECC length distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['text_length'], bins=30, kde=True)
    plt.title('Distribution of ECC Lengths')
    plt.xlabel('ECC Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'ecc_length_distribution.png')
    plt.savefig(plot_path)
    plt.close()

def plot_ecc_length_by_company(results_df, top_n=20):
    print("Computing ECC length by company...")
    top_companies = results_df['company_info'].value_counts().nlargest(top_n).index
    top_results_df = results_df[results_df['company_info'].isin(top_companies)]

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='company_info', y='text_length', data=top_results_df)
    plt.title(f'ECC Length Distribution by Top {top_n} Companies')
    plt.xlabel('Company')
    plt.ylabel('ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'ecc_length_by_company.png')
    plt.savefig(plot_path)
    plt.close()

def plot_ecc_length_over_time(results_df):
    print("Computing ECC length over time...")
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values(by='date')

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='date', y='text_length', data=results_df)
    plt.title('ECC Length Over Time')
    plt.xlabel('Date')
    plt.ylabel('ECC Length (Number of Words)')
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'ecc_length_over_time.png')
    plt.savefig(plot_path)
    plt.close()

def plot_files_distribution(results_df):
    print("Computing files distribution...")
    file_counts = results_df['permco'].value_counts()

    plt.figure(figsize=(12, 8))
    sns.histplot(file_counts, bins=30, kde=True)
    plt.title('Distribution of Files per Permco')
    plt.xlabel('Number of Files')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'files_distribution.png')
    plt.savefig(plot_path)
    plt.close()

def plot_average_ecc_length_per_company(results_df):
    print("Computing average ECC length per company...")
    avg_length_per_company = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    avg_length_per_company.plot(kind='bar')
    plt.title('Average ECC Length by Top 20 Companies')
    plt.xlabel('Company')
    plt.ylabel('Average ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'average_ecc_length_per_company.png')
    plt.savefig(plot_path)
    plt.close()

def plot_ecc_length_distribution_by_year(results_df):
    print("Computing ECC length distribution by year...")
    results_df['year'] = pd.to_datetime(results_df['date']).dt.year  # Convert date to year
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='year', y='text_length', data=results_df)
    plt.title('ECC Length Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('ECC Length (Number of Words)')
    plt.xticks(rotation=90)
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(ecc_plots_folder, 'ecc_length_distribution_by_year.png')
    plt.savefig(plot_path)
    plt.close()


def additional_descriptive_statistics(results_df):
    print("Computing additional descriptive statistics...")
    # Calculate the number of unique companies
    num_unique_companies = results_df['company_info'].nunique()

    # Calculate calls per company
    calls_per_company = results_df['company_info'].value_counts().to_frame().reset_index()
    calls_per_company.columns = ['Company', 'Number of Calls']

    # Calculate the top 5 companies by average ECC length
    top5_avg_length = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(5).to_frame().reset_index()
    top5_avg_length.columns = ['Company', 'Average ECC Length']

    # Create summary statistics table
    summary_stats = results_df.describe().transpose()
    summary_stats_table = pd.DataFrame(summary_stats)

    return num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table


def display_tables(num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table, output_html_path):
    # Display tables in HTML
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

    # Save HTML content to file
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
    ecc_sample = load_data(file_handler, sample_size, random_seed)
    results_df = convert_to_dataframe(ecc_sample)

    # Plotting different statistics
    plot_paragraph_length_distribution(text_processor, results_df)
    plot_ecc_length_distribution(results_df)
    plot_ecc_length_by_company(results_df)
    plot_ecc_length_over_time(results_df)
    plot_files_distribution(results_df)
    plot_average_ecc_length_per_company(results_df)
    plot_ecc_length_distribution_by_year(results_df)  # Ensure this is included

    # Add NLP-related plots
    plot_bag_of_words(results_df)
    plot_wordcloud(results_df)
    #plot_tfidf_top_terms(results_df)
    plot_topics_tsne_pca(results_df)
    #plot_ngram_frequencies(results_df, n=2)  # Bigrams
    #plot_ngram_frequencies(results_df, n=3)  # Trigrams
    plot_sentiment_analysis(results_df)
    plot_keyword_cooccurrence(results_df)
    plot_word_length_distribution(results_df)
    #plot_pos_tagging_distribution(results_df)
    #plot_ner(results_df)

    # Descriptive statistics and saving tables
    num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table = additional_descriptive_statistics(results_df)
    output_html_path = os.path.join(ecc_plots_folder, 'ecc_statistics.html')  # Save in ecc_plots_folder
    display_tables(num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table, output_html_path)

    print(f"All plots and reports have been saved in: {ecc_plots_folder}")


if __name__ == "__main__":
    main()
