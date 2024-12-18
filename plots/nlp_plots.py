import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
from textblob import TextBlob
import pandas as pd
import json
import os

class NLPPlotter:
    def __init__(self, config_path):
        """
        Initialize the NLPPlotter class.

        Args:
            config_path (str): Path to the configuration file (JSON format) containing
                the ECC plots folder path and any other necessary parameters.
        """
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)
        self.ecc_plots_folder = self.config["ecc_plots_folder"]
        self.nlp = spacy.load("en_core_web_sm")

    def plot_tfidf_top_terms(self, results_df, top_n=20):
        """
        Plot the top TF-IDF terms for the given dataframe.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.
            top_n (int, optional): Number of top terms to plot. Defaults to 20.

        Returns:
            None
        """
        print("Computing TF-IDF...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        X = vectorizer.fit_transform(results_df['text'])
        tfidf_scores = np.asarray(X.mean(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()

        term_scores = sorted(zip(terms, tfidf_scores), key=lambda x: x[1], reverse=True)
        sorted_terms = [ts[0] for ts in term_scores]
        sorted_scores = [ts[1] for ts in term_scores]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=sorted_scores, y=sorted_terms)
        plt.title(f'Top {top_n} TF-IDF Terms')
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Terms')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'tfidf_top_terms.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_topics_tsne_pca(self, results_df):
        """
        Plot the results of applying PCA and t-SNE to the topics extracted from the ECC.

        The function will compute the PCA and t-SNE transformations of the topics in the given dataframe
        and then plot the results using seaborn's scatterplot function. The resulting plots will be saved
        in the ECC plots folder specified in the configuration file.

        Args:
            results_df (pd.DataFrame): Dataframe containing the topics to be analyzed.

        Returns:
            None
        """
        print("Computing PCA and t-SNE...")
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(results_df['text'])

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(X.toarray())

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(X.toarray())

        results_df['pca_one'] = pca_results[:,0]
        results_df['pca_two'] = pca_results[:,1]
        results_df['tsne_one'] = tsne_results[:,0]
        results_df['tsne_two'] = tsne_results[:,1]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='pca_one', y='pca_two', data=results_df, hue='company_info', palette='viridis')
        plt.title('PCA of Topics')
        plot_path_pca = os.path.join(self.ecc_plots_folder, 'pca_topics.png')
        plt.savefig(plot_path_pca)
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_one', y='tsne_two', data=results_df, hue='company_info', palette='viridis')
        plt.title('t-SNE of Topics')
        plot_path_tsne = os.path.join(self.ecc_plots_folder, 'tsne_topics.png')
        plt.savefig(plot_path_tsne)
        plt.close()

    def plot_ngram_frequencies(self, results_df, n=2, top_n=20):
        """
        Plot the top N-gram frequencies for the given dataframe.

        This function uses a CountVectorizer to extract and count the N-grams
        from the text data in the dataframe. It then plots the top N most
        frequent N-grams using a bar plot.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.
            n (int, optional): The number of words in the N-grams. Defaults to 2 (bigrams).
            top_n (int, optional): Number of top N-grams to plot. Defaults to 20.

        Returns:
            None
        """

        print(f"Creating {n}-gram plot...")
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english', max_features=top_n)
        X = vectorizer.fit_transform(results_df['text'])
        ngram_counts = np.asarray(X.sum(axis=0)).flatten()
        ngrams = vectorizer.get_feature_names_out()

        ngram_scores = sorted(zip(ngrams, ngram_counts), key=lambda x: x[1], reverse=True)
        sorted_ngrams = [ns[0] for ns in ngram_scores]
        sorted_counts = [ns[1] for ns in ngram_scores]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=sorted_counts, y=sorted_ngrams)
        plt.title(f'Top {top_n} {"Bigrams" if n == 2 else "Trigrams"}')
        plt.xlabel('Frequency')
        plt.ylabel('N-grams')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, f'ngram_frequencies_{n}.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_sentiment_analysis(self, results_df):
        """
        Plot the sentiment analysis of the given dataframe.

        This function uses the TextBlob library to calculate the sentiment
        polarity of each text in the dataframe. It then plots the distribution
        of sentiment polarities using a histogram.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.

        Returns:
            None
        """
        print("Computing sentiment analysis...")
        results_df['sentiment'] = results_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['sentiment'], bins=30, kde=True)
        plt.title('Sentiment Analysis')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'sentiment_analysis.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_keyword_cooccurrence(self, results_df, top_n=20):
        """
        Plot a keyword co-occurrence network for the given dataframe.

        This function uses a CountVectorizer to extract the top N keywords from the text data
        in the dataframe and computes a co-occurrence matrix. It then creates a network graph
        using NetworkX to visualize the relationships between these keywords.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.
            top_n (int, optional): Number of top keywords to consider for the co-occurrence network. Defaults to 20.

        Returns:
            None
        """

        print("Computing keyword co-occurrence network...")
        vectorizer = CountVectorizer(max_features=top_n, stop_words='english')
        X = vectorizer.fit_transform(results_df['text'])
        terms = vectorizer.get_feature_names_out()

        cooccurrence_matrix = (X.T * X)
        cooccurrence_matrix.setdiag(0)
        cooccurrence_df = pd.DataFrame(cooccurrence_matrix.toarray(), index=terms, columns=terms)

        G = nx.from_pandas_adjacency(cooccurrence_df)

        plt.figure(figsize=(12, 12), facecolor='white')
        pos = nx.spring_layout(G, k=0.3)
        nx.draw_networkx_edges(G, pos, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="skyblue", alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
        plt.title('Keyword Co-occurrence Network')
        plot_path = os.path.join(self.ecc_plots_folder, 'keyword_cooccurrence_network.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_word_length_distribution(self, results_df):
        """
        Plot the distribution of word lengths in the given dataframe.

        This function uses a list comprehension to extract the word lengths from
        the text data in the dataframe and plots the distribution of word lengths
        using a histogram.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.

        Returns:
            None
        """

        print("Computing word length distribution...")
        word_lengths = [len(word) for text in results_df['text'] for word in text.split()]

        plt.figure(figsize=(10, 6))
        sns.histplot(word_lengths, bins=30, kde=True)
        plt.title('Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'word_length_distribution.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_bag_of_words(self, results_df):
        """
        Plot the top 20 words in the given dataframe using a barplot.

        This function uses CountVectorizer to compute the word counts in the
        given dataframe and plots the top 20 words using a barplot.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.

        Returns:
            None
        """
        print("Computing bag of words...")
        combined_text = " ".join(results_df['text'].tolist())
        vectorizer = CountVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform([combined_text])
        words = vectorizer.get_feature_names_out()
        counts = X.toarray().flatten()

        word_counts = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
        sorted_words = [wc[0] for wc in word_counts]
        sorted_counts = [wc[1] for wc in word_counts]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=sorted_counts, y=sorted_words)
        plt.title('Top 20 Words in ECCs')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'bag_of_words.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_wordcloud(self, results_df):
        """
        Generate and plot a word cloud from the text data in the given dataframe.

        This function combines text data from the dataframe, creates a word cloud
        using the WordCloud library, and saves the generated plot as an image.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.

        Returns:
            None
        """

        print("Computing wordcloud...")
        chunk_size = 1000
        combined_text = " ".join(results_df['text'].iloc[:chunk_size].tolist())

        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(combined_text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of ECCs')
        plot_path = os.path.join(self.ecc_plots_folder, 'wordcloud.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_ner(self, results_df, chunk_size=50):
        """
        Analyze and plot the distribution of named entities in the text data.

        This function processes chunks of text data from the given dataframe using
        a pre-trained NLP model to perform named entity recognition (NER). It
        counts the frequency of each named entity type and visualizes the distribution
        using a bar plot.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.
            chunk_size (int, optional): Number of text entries to process in each chunk.
                                        Defaults to 50.

        Returns:
            None
        """

        print("Computing named entity recognition (NER)...")
        all_ents = []
        num_chunks = len(results_df) // chunk_size + 1

        for i in range(num_chunks):
            chunk = results_df['text'].iloc[i * chunk_size:(i + 1) * chunk_size]
            for doc in self.nlp.pipe(chunk, batch_size=10):
                all_ents.extend([ent.label_ for ent in doc.ents])

        ent_counts = Counter(all_ents)
        ent_labels = list(ent_counts.keys())
        ent_values = list(ent_counts.values())

        plt.figure(figsize=(12, 8))
        sns.barplot(x=ent_values, y=ent_labels)
        plt.title('Named Entity Recognition (NER) Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('Entity Type')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'ner_distribution.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_pos_tagging_distribution(self, results_df):
        """
        Analyze and plot the distribution of POS tags in the text data.

        This function processes chunks of text data from the given dataframe using
        a pre-trained NLP model to perform POS tagging. It counts the frequency of
        each POS tag and visualizes the distribution using a bar plot.

        Args:
            results_df (pd.DataFrame): Dataframe containing the text data to be analyzed.

        Returns:
            None
        """
        print("Computing POS tagging distribution...")
        pos_tags = [token.pos_ for doc in self.nlp.pipe(results_df['text'], batch_size=50) for token in doc]
        pos_counts = Counter(pos_tags)

        pos_labels = list(pos_counts.keys())
        pos_values = list(pos_counts.values())

        plt.figure(figsize=(12, 8))
        sns.barplot(x=pos_values, y=pos_labels)
        plt.title('POS Tagging Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('POS Tag')
        plt.grid(True)
        plot_path = os.path.join(self.ecc_plots_folder, 'pos_tagging_distribution.png')
        plt.savefig(plot_path)
        plt.close()
