# evaluate_topics.py

import os
import random
import json
import pandas as pd
import numpy as np  # Add numpy import

def generate_evaluation_file(topic_model, results_df, output_dir, text_column, topics_column, section_type):
    """
    Generate a human evaluation file with topic information and document-topic pairs for a specific section,
    ensuring at least one document from each of the top 10 topics and using weighted sampling for the rest.

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
        # Parse JSON lists
        non_outlier_df['topics_list'] = non_outlier_df[topics_column].apply(json.loads)
        non_outlier_df['sections_list'] = non_outlier_df[text_column].apply(json.loads)

        # Build list of documents with their topics
        non_outlier_docs = []
        for idx, row in non_outlier_df.iterrows():
            sections = row['sections_list']
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic != -1:
                    non_outlier_docs.append({'doc': doc, 'topic': topic})

        if not non_outlier_docs:
            print(f"No non-outlier {section_type} documents available for sampling.")
        else:
            # Create a DataFrame from non_outlier_docs
            non_outlier_docs_df = pd.DataFrame(non_outlier_docs)

            # Get topic counts
            topic_counts = non_outlier_docs_df['topic'].value_counts().to_dict()

            # Identify top 10 topics
            top_10_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_10_topic_ids = [topic for topic, count in top_10_topics]
            print(f"Top 10 Topics for {section_type}: {top_10_topic_ids}")

            # Sample one document from each of the top 10 topics
            top_topic_samples = []
            for topic_id in top_10_topic_ids:
                topic_docs = non_outlier_docs_df[non_outlier_docs_df['topic'] == topic_id]
                if not topic_docs.empty:
                    sampled_doc = topic_docs.sample(n=2, random_state=42).iloc[0].to_dict()
                    top_topic_samples.append(sampled_doc)
                else:
                    print(f"No documents found for Top Topic ID {topic_id} in {section_type}.")

            # Remove sampled documents from the pool to avoid duplication
            sampled_docs_set = set([d['doc'] for d in top_topic_samples])
            remaining_docs_df = non_outlier_docs_df[~non_outlier_docs_df['doc'].isin(sampled_docs_set)]

            # Calculate weights for remaining documents based on topic counts
            if not remaining_docs_df.empty:
                remaining_topic_counts = remaining_docs_df['topic'].map(topic_counts)
                probabilities = remaining_topic_counts / remaining_topic_counts.sum()

                # Determine number of remaining samples
                total_samples = 30
                top_samples_count = len(top_topic_samples)
                remaining_samples_count = total_samples - top_samples_count

                remaining_samples_count = max(0, remaining_samples_count)  # Ensure non-negative

                # Adjust if not enough remaining samples
                actual_remaining_samples = min(remaining_samples_count, len(remaining_docs_df))
                if actual_remaining_samples > 0:
                    sampled_remaining_docs = remaining_docs_df.sample(
                        n=actual_remaining_samples, 
                        replace=False, 
                        weights=probabilities, 
                        random_state=42
                    ).to_dict('records')
                else:
                    sampled_remaining_docs = []
            else:
                sampled_remaining_docs = []
                print(f"No remaining non-outlier {section_type} documents available for weighted sampling.")

            # Combine top topic samples with remaining samples
            final_sampled_docs = top_topic_samples + sampled_remaining_docs

            # Write sampled documents to the evaluation file
            if final_sampled_docs:
                f.write(f"Selected Document-Topic Pairs for {section_type}:\n\n")
                for item in final_sampled_docs:
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
            else:
                f.write(f"No non-outlier {section_type} documents available for sampling.\n\n")

        # Now sample outlier documents
        outlier_df = results_df.copy()
        outlier_df = outlier_df.dropna(subset=[topics_column, text_column])
        outlier_df['topics_list'] = outlier_df[topics_column].apply(json.loads)
        outlier_df['sections_list'] = outlier_df[text_column].apply(json.loads)

        outlier_docs = []
        for idx, row in outlier_df.iterrows():
            sections = row['sections_list']
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic == -1:
                    outlier_docs.append({'doc': doc, 'topic': topic})

        # Randomly sample 20 outlier documents
        num_outlier_samples = min(20, len(outlier_docs))
        if num_outlier_samples == 0:
            print(f"No outlier {section_type} documents available for sampling.")
        else:
            random.seed(42)
            sampled_outliers = random.sample(outlier_docs, num_outlier_samples)

            f.write(f"Random Outlier Document-Topic Pairs for {section_type}:\n")
            for item in sampled_outliers:
                doc = item['doc']
                f.write(f"Document: {doc}\n")
                f.write(f"Topic ID: -1 (Outlier)\n\n")
