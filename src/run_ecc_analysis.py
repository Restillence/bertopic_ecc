from ecc_data_exploration import ECCDataExplorer
from nlp_plots import NLPPlotter
import os
import torch

def main():
    # Initialize ECCDataExplorer
    ecc_explorer = ECCDataExplorer(config_path="config.json")
    
    # Load data and convert to dataframe
    ecc_sample = ecc_explorer.load_data()
    results_df = ecc_explorer.convert_to_dataframe(ecc_sample)
    
    # Perform data exploration and plotting
    ecc_explorer.plot_paragraph_length_distribution(results_df)
    ecc_explorer.plot_ecc_length_distribution(results_df)
    ecc_explorer.plot_ecc_length_by_company(results_df)
    ecc_explorer.plot_ecc_length_over_time(results_df)
    ecc_explorer.plot_files_distribution(results_df)
    ecc_explorer.plot_average_ecc_length_per_company(results_df)
    ecc_explorer.plot_ecc_length_distribution_by_year(results_df)

    # Add new plot for average paragraph length distribution
    ecc_explorer.plot_avg_paragraph_length_distribution(results_df)

    # Display additional stats
    num_unique_companies, calls_per_company, top5_avg_length, summary_stats_table = ecc_explorer.additional_descriptive_statistics(results_df)
    ecc_explorer.display_tables(
        num_unique_companies, 
        calls_per_company, 
        top5_avg_length, 
        summary_stats_table, 
        os.path.join(ecc_explorer.ecc_plots_folder, 'ecc_statistics.html')
    )

    # Initialize NLPPlotter
    nlp_plotter = NLPPlotter(config_path="config.json")
    
    # Generate NLP-related plots
    nlp_plotter.plot_tfidf_top_terms(results_df)
    nlp_plotter.plot_topics_tsne_pca(results_df)
    nlp_plotter.plot_sentiment_analysis(results_df)
    nlp_plotter.plot_keyword_cooccurrence(results_df)
    nlp_plotter.plot_word_length_distribution(results_df)
    nlp_plotter.plot_bag_of_words(results_df)
    nlp_plotter.plot_wordcloud(results_df)

if __name__ == "__main__":
    main()