import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from file_handling import FileHandler
from text_processing import TextProcessor

class ECCDataExplorer:
    def __init__(self, config_path):
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)
        
        self.folderpath_ecc = self.config["folderpath_ecc"]
        self.index_file_ecc_folder = self.config["index_file_ecc_folder"]
        self.ecc_plots_folder = self.config["ecc_plots_folder"]
        self.sample_size = self.config["sample_size"]
        self.random_seed = self.config["random_seed"]
        self.index_file_path = os.path.join(self.index_file_ecc_folder, self.config["index_file_path"])
        
        # Initializing necessary objects
        self.text_processor = TextProcessor(method=self.config["document_split"], section_to_analyze=self.config["section_to_analyze"])
        self.file_handler = FileHandler(index_file_path=self.index_file_path, folderpath_ecc=self.folderpath_ecc)

    def load_data(self):
        np.random.seed(self.random_seed)
        return self.file_handler.create_ecc_sample(self.sample_size)

    def convert_to_dataframe(self, ecc_sample):
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

    def plot_paragraph_length_distribution(self, results_df):
        print("Computing paragraph word length distribution...")
        paragraph_lengths = []

        for _, row in results_df.iterrows():
            paragraphs = self.text_processor.split_text(row['text'])
            paragraph_lengths.extend([len(paragraph.split()) for paragraph in paragraphs])

        plt.figure(figsize=(10, 6))
        sns.histplot(paragraph_lengths, bins=30, kde=True)
        plt.title('Distribution of Paragraph Word Lengths')
        plt.xlabel('Paragraph Length (Number of Words)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'paragraph_word_length_distribution.png'))
        plt.close()

    def plot_ecc_length_distribution(self, results_df):
        print("Computing ECC length distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['text_length'], bins=30, kde=True)
        plt.title('Distribution of ECC Lengths')
        plt.xlabel('ECC Length (Number of Words)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'ecc_length_distribution.png'))
        plt.close()

    def plot_ecc_length_by_company(self, results_df, top_n=20):
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
        plt.savefig(os.path.join(self.ecc_plots_folder, 'ecc_length_by_company.png'))
        plt.close()

    def plot_ecc_length_over_time(self, results_df):
        print("Computing ECC length over time...")
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.sort_values(by='date')

        plt.figure(figsize=(12, 8))
        sns.lineplot(x='date', y='text_length', data=results_df)
        plt.title('ECC Length Over Time')
        plt.xlabel('Date')
        plt.ylabel('ECC Length (Number of Words)')
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'ecc_length_over_time.png'))
        plt.close()

    def plot_files_distribution(self, results_df):
        print("Computing files distribution...")
        file_counts = results_df['permco'].value_counts()

        plt.figure(figsize=(12, 8))
        sns.histplot(file_counts, bins=30, kde=True)
        plt.title('Distribution of Files per Permco')
        plt.xlabel('Number of Files')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'files_distribution.png'))
        plt.close()

    def plot_average_ecc_length_per_company(self, results_df):
        print("Computing average ECC length per company...")
        avg_length_per_company = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        avg_length_per_company.plot(kind='bar')
        plt.title('Average ECC Length by Top 20 Companies')
        plt.xlabel('Company')
        plt.ylabel('Average ECC Length (Number of Words)')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'average_ecc_length_per_company.png'))
        plt.close()

    def plot_ecc_length_distribution_by_year(self, results_df):
        print("Computing ECC length distribution by year...")
        results_df['year'] = pd.to_datetime(results_df['date']).dt.year  # Convert date to year
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='year', y='text_length', data=results_df)
        plt.title('ECC Length Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('ECC Length (Number of Words)')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.savefig(os.path.join(self.ecc_plots_folder, 'ecc_length_distribution_by_year.png'))
        plt.close()

    def additional_descriptive_statistics(self, results_df):
        print("Computing additional descriptive statistics...")
        num_unique_companies = results_df['company_info'].nunique()
        calls_per_company = results_df['company_info'].value_counts().to_frame().reset_index()
        calls_per_company.columns = ['Company', 'Number of Calls']
        top5_avg_length = results_df.groupby('company_info')['text_length'].mean().sort_values(ascending=False).head(5).to_frame().reset_index()
        top5_avg_length.columns = ['Company', 'Average ECC Length']
        summary_stats_table = results_df.describe().transpose()

        return num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table

    def display_tables(self, num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table, output_html_path):
        print("Displaying and saving tables...")
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
        print(calls_per_company.head(10).to_string(index=False))
        print("\nTop 5 companies by average ECC length:")
        print(top5_avg_length.to_string(index=False))
        print(f"\nTables have been saved to {output_html_path}")
