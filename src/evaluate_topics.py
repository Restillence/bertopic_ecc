import os
import random
import json
import pandas as pd

def generate_evaluation_file(topic_model, results_df, output_dir):
    """
    Generate a human evaluation file with topic information and random document-topic pairs.

    Args:
        topic_model: The trained BERTopic model.
        results_df: DataFrame containing the documents and their assigned topics.
        output_dir: The directory where the evaluation file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'human_eval_topics_docs.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        # First, print topic information
        topic_info = topic_model.get_topic_info()
        f.write("Topic Information:\n")
        f.write(topic_info[['Topic', 'Name', 'Count']].to_string(index=False))
        f.write("\n\n")

        # For each topic, get the representation
        f.write("Topic Representations:\n")
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue  # Skip outliers
            representation = topic_model.get_topic(topic_id)
            words = ', '.join([word for word, _ in representation])
            f.write(f"Topic {topic_id} Representation: {words}\n")
        f.write("\n")

        # Exclude outliers
        non_outlier_df = results_df.copy()
        non_outlier_df['topics_list'] = non_outlier_df['topics'].apply(json.loads)
        non_outlier_docs = []
        for idx, row in non_outlier_df.iterrows():
            sections = json.loads(row['text'])
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic != -1:
                    non_outlier_docs.append({'doc': doc, 'topic': topic})

        # Randomly sample 30 documents
        random.seed(42)
        num_samples = min(30, len(non_outlier_docs))
        if num_samples == 0:
            print("No non-outlier documents available for sampling.")
        else:
            sampled_docs = random.sample(non_outlier_docs, num_samples)

            f.write("Random Document-Topic Pairs:\n")
            for item in sampled_docs:
                doc = item['doc']
                topic_id = item['topic']
                # Get topic info
                topic_name = topic_model.get_topic(topic_id)
                words = ', '.join([word for word, _ in topic_name])
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
        outlier_df['topics_list'] = outlier_df['topics'].apply(json.loads)
        outlier_docs = []
        for idx, row in outlier_df.iterrows():
            sections = json.loads(row['text'])
            section_topics = row['topics_list']
            for doc, topic in zip(sections, section_topics):
                if topic == -1:
                    outlier_docs.append({'doc': doc, 'topic': topic})

        # Randomly sample 20 outlier documents
        num_outlier_samples = min(20, len(outlier_docs))
        if num_outlier_samples == 0:
            print("No outlier documents available for sampling.")
        else:
            sampled_outliers = random.sample(outlier_docs, num_outlier_samples)

            f.write("Random Outlier Document-Topic Pairs:\n")
            for item in sampled_outliers:
                doc = item['doc']
                f.write(f"Document: {doc}\n")
                f.write(f"Topic ID: -1 (Outlier)\n\n")
