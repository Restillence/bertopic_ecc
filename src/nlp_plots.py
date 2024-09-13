#nlp_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import spacy
from collections import Counter
import networkx as nx
from textblob import TextBlob
import pandas as pd
import json
import os

# Load the config.json to get ecc_plots_folder
with open("config.json", "r") as config_file:
    config = json.load(config_file)

ecc_plots_folder = config["ecc_plots_folder"]

# Load Spacy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function Definitions

def plot_ner(results_df, chunk_size=50):
    print("Computing named entity recognition (NER)...")
    all_ents = []
    num_chunks = len(results_df) // chunk_size + 1

    for i in range(num_chunks):
        chunk = results_df['text'].iloc[i * chunk_size:(i + 1) * chunk_size]
        for doc in nlp.pipe(chunk, batch_size=10):
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

    # Save NER plot
    plot_path = os.path.join(ecc_plots_folder, 'ner_distribution.png')
    plt.savefig(plot_path)
    plt.close()


def plot_tfidf_top_terms(results_df, top_n=20):
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

    # Save and close plot
    plot_path = os.path.join(ecc_plots_folder, 'tfidf_top_terms.png')
    plt.savefig(plot_path)
    plt.close()


def plot_topics_tsne_pca(results_df):
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

    # PCA Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca_one', y='pca_two', data=results_df, hue='company_info', palette='viridis')
    plt.title('PCA of Topics')

    # Save and close PCA plot
    plot_path_pca = os.path.join(ecc_plots_folder, 'pca_topics.png')
    plt.savefig(plot_path_pca)
    plt.close()

    # t-SNE Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tsne_one', y='tsne_two', data=results_df, hue='company_info', palette='viridis')
    plt.title('t-SNE of Topics')

    # Save and close t-SNE plot
    plot_path_tsne = os.path.join(ecc_plots_folder, 'tsne_topics.png')
    plt.savefig(plot_path_tsne)
    plt.close()


def plot_ngram_frequencies(results_df, n=2, top_n=20):
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

    # Save and close n-gram plot
    plot_path = os.path.join(ecc_plots_folder, f'ngram_frequencies_{n}.png')
    plt.savefig(plot_path)
    plt.close()


def plot_sentiment_analysis(results_df):
    print("Computing sentiment analysis...")
    results_df['sentiment'] = results_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['sentiment'], bins=30, kde=True)
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save and close sentiment analysis plot
    plot_path = os.path.join(ecc_plots_folder, 'sentiment_analysis.png')
    plt.savefig(plot_path)
    plt.close()


def plot_keyword_cooccurrence(results_df, top_n=20):
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

    # Save and close keyword co-occurrence plot
    plot_path = os.path.join(ecc_plots_folder, 'keyword_cooccurrence_network.png')
    plt.savefig(plot_path)
    plt.close()


def plot_word_length_distribution(results_df):
    print("Computing word length distribution...")
    word_lengths = [len(word) for text in results_df['text'] for word in text.split()]

    plt.figure(figsize=(10, 6))
    sns.histplot(word_lengths, bins=30, kde=True)
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save and close word length distribution plot
    plot_path = os.path.join(ecc_plots_folder, 'word_length_distribution.png')
    plt.savefig(plot_path)
    plt.close()


def plot_pos_tagging_distribution(results_df):
    print("Computing POS tagging distribution...")
    pos_tags = [token.pos_ for doc in nlp.pipe(results_df['text'], batch_size=50) for token in doc]
    pos_counts = Counter(pos_tags)

    pos_labels = list(pos_counts.keys())
    pos_values = list(pos_counts.values())

    plt.figure(figsize=(12, 8))
    sns.barplot(x=pos_values, y=pos_labels)
    plt.title('POS Tagging Distribution')
    plt.xlabel('Frequency')
    plt.ylabel('POS Tag')
    plt.grid(True)

    # Save and close POS tagging plot
    plot_path = os.path.join(ecc_plots_folder, 'pos_tagging_distribution.png')
    plt.savefig(plot_path)
    plt.close()


def plot_bag_of_words(results_df):
    print("Computing bag of words...")
    # Combine all texts
    combined_text = " ".join(results_df['text'].tolist())

    # Create a CountVectorizer
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    X = vectorizer.fit_transform([combined_text])
    words = vectorizer.get_feature_names_out()
    counts = X.toarray().flatten()

    # Sort words by count
    word_counts = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
    sorted_words = [wc[0] for wc in word_counts]
    sorted_counts = [wc[1] for wc in word_counts]

    # Plot Bag of Words
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_counts, y=sorted_words)
    plt.title('Top 20 Words in ECCs')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.grid(True)

    # Save and close bag of words plot
    plot_path = os.path.join(ecc_plots_folder, 'bag_of_words.png')
    plt.savefig(plot_path)
    plt.close()


def plot_wordcloud(results_df):
    print("Computing wordcloud...")
    # Combine all texts in chunks to manage memory
    chunk_size = 1000
    combined_text = " ".join(results_df['text'].iloc[:chunk_size].tolist())

    # Create and plot WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(combined_text)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of ECCs')

    # Save and close word cloud plot
    plot_path = os.path.join(ecc_plots_folder, 'wordcloud.png')
    plt.savefig(plot_path)
    plt.close()
