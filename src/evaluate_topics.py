# evaluate_topics.py

import os
import random
import json
import pandas as pd

def generate_evaluation_file(topic_model, results_df, output_dir, text_column, topics_column, section_type):
    """
    Generate a human evaluation file with topic information and random document-topic pairs for a specific section.

    Args:
        topic_model: The trained BERTopic model.
        results_df: DataFrame containing the documents and their assigned topics.
        output_dir: The directory where the evaluation file will be saved.
        text_column: The column name for the document texts.
        topics_column: The column name for the assigned topics.
        section_type: 'Presentation' or 'Q&A' to differentiate the evaluation file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'human_eval_{section_type}_topics_docs.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        # First, print topic information
        topic_info = topic_model.get_topic_info()
        f.write(f"Topic Information for {section_type}:\n")
        f.write(topic_info[['Topic', 'Name', 'Count']].to_string(index=False))
        f.write("\n\n")

        # For each topic, get the representation
        f.write(f"Topic Representations for {section_type}:\n")
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue  # Skip outliers
            representation = topic_model.get_topic(topic_id)
            words = ', '.join([word for word, _ in representation])
            f.write(f"Topic {topic_id} Representation: {words}\n")
        f.write("\n")

        # Exclude outliers
        non_outlier_df = results_df.copy()
        # Ensure the topics_column is not empty or NaN
        non_outlier_df = non_outlier_df.dropna(subset=[topics_column, text_column])
        non_outlier_df['topics_list'] = non_outlier_df[topics_column].apply(json.loads)
        non_outlier_docs = []
        for idx, row in non_outlier_df.iterrows():
            sections = json.loads(row[text_column])
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic != -1:
                    non_outlier_docs.append({'doc': doc, 'topic': topic})

        # Randomly sample 30 documents
        random.seed(42)
        num_samples = min(30, len(non_outlier_docs))
        if num_samples == 0:
            print(f"No non-outlier {section_type} documents available for sampling.")
        else:
            sampled_docs = random.sample(non_outlier_docs, num_samples)

            f.write(f"Random Document-Topic Pairs for {section_type}:\n")
            for item in sampled_docs:
                doc = item['doc']
                topic_id = item['topic']
                # Get topic info
                representation = topic_model.get_topic(topic_id)
                words = ', '.join([word for word, _ in representation])
                topic_info_row = topic_info[topic_info['Topic'] == topic_id]
                count = topic_info_row['Count'].values[0] if not topic_info_row.empty else 'N/A'
                name = topic_info_row['Name'].values[0] if not topic_info_row.empty else 'N/A'
                f.write(f"Document: {doc}\n")
                f.write(f"Topic ID: {topic_id}\n")
                f.write(f"Topic Name: {name}\n")
                f.write(f"Topic Count: {count}\n")
                f.write(f"Topic Representation: {words}\n\n")

        # Now sample outlier documents
        outlier_df = results_df.copy()
        outlier_df = outlier_df.dropna(subset=[topics_column, text_column])
        outlier_df['topics_list'] = outlier_df[topics_column].apply(json.loads)
        outlier_docs = []
        for idx, row in outlier_df.iterrows():
            sections = json.loads(row[text_column])
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic == -1:
                    outlier_docs.append({'doc': doc, 'topic': topic})

        # Randomly sample 20 outlier documents
        num_outlier_samples = min(20, len(outlier_docs))
        if num_outlier_samples == 0:
            print(f"No outlier {section_type} documents available for sampling.")
        else:
            sampled_outliers = random.sample(outlier_docs, num_outlier_samples)

            f.write(f"Random Outlier Document-Topic Pairs for {section_type}:\n")
            for item in sampled_outliers:
                doc = item['doc']
                f.write(f"Document: {doc}\n")
                f.write(f"Topic ID: -1 (Outlier)\n\n")

    # The main function is not needed as this script is intended to be imported
