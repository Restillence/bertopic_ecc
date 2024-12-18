# bertopic_fitting_visus.py

# Import necessary libraries
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder  # Import BaseEmbedder for custom embedding
from file_handling import FileHandler
from text_processing import TextProcessor
from utils import print_configuration, count_items  # Import count_items instead of count_word_length_text
from sentence_transformers import SentenceTransformer
from scipy.cluster import hierarchy as sch  # For hierarchical topic modeling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # You can add a FileHandler here to log to a file if desired
    ]
)

# Adjust the path to include 'src' if it's not already in the system path
current_dir = os.getcwd()
if "src" not in current_dir:
    src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
    sys.path.append(src_path)

# Import the evaluation module
from evaluate_topics import generate_evaluation_file  # Import the evaluation function

class CustomEmbeddingModel(BaseEmbedder):
    """
    A wrapper class for the SentenceTransformer model to add the 'embed_documents' method
    expected by BERTopic.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, documents, verbose=False):
        return self.embedding_model.encode(documents, show_progress_bar=verbose)

    def embed_queries(self, queries, verbose=False):
        return self.embedding_model.encode(queries, show_progress_bar=verbose)

class BertopicFitting:
    """
    Class to fit BERTopic models, save results, and generate visualizations for a given ECC sample.
    """

    def __init__(self, config, model_load_path):
        """
        Initialize the BertopicFitting class.

        Args:
            config (dict): Configuration dictionary containing parameters for the BERTopic model.
            model_load_path (str): Filepath to the pre-trained BERTopic model.
        """
        self.config = config
        self.model_load_path = model_load_path
        self.index_file_ecc_folder = config["index_file_ecc_folder"]
        self.output_dir = "model_outputs"
        self.eval_dir = "eval"  # Define evaluation directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)  # Create eval directory

        # Load the embedding model and wrap it
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(self.config["embedding_model_choice"], device=self.device)
        logging.info(f"Embedding model loaded successfully. Using device: {self.device}")

        # Wrap the embedding model
        self.embedding_model = CustomEmbeddingModel(embedding_model)

        # Load the BERTopic model with the custom embedding model
        self.topic_model = self._load_bertopic_model()
        # Removed: self.topic_model.embedding_model = self.embedding_model

    def _load_bertopic_model(self):
        """
        Load the pre-trained BERTopic model from the given filepath.
        """
        logging.info(f"Loading BERTopic model from {self.model_load_path}...")
        # Load the BERTopic model with the embedding model
        topic_model = BERTopic.load(self.model_load_path, embedding_model=self.embedding_model)
        return topic_model

    def save_results(self, all_relevant_sections, topics_sections, all_relevant_questions, topics_questions, all_management_answers, topics_answers, ecc_sample):
        """
        Save the results to a CSV file and store the DataFrame.
        """
        # Initialize a dictionary to store the results
        result_dict = {}
        topic_idx_sections = 0
        topic_idx_questions = 0
        topic_idx_answers = 0
        # Iterate over the sample and save the results
        for permco, calls in ecc_sample.items():
            # Iterate over each call
            for call_id, value in calls.items():
                sections = value.get('presentation_text', [])
                num_sections = len(sections)
                section_topics = topics_sections[topic_idx_sections:topic_idx_sections + num_sections]

                questions = value.get('participant_questions', [])
                num_questions = len(questions)
                question_topics = topics_questions[topic_idx_questions:topic_idx_questions + num_questions]

                answers = value.get('management_answers', [])
                num_answers = len(answers)
                answer_topics = topics_answers[topic_idx_answers:topic_idx_answers + num_answers]

                # Ensure topics and text lists have the same length
                if len(section_topics) != num_sections:
                    raise ValueError(f"Mismatch between number of topics and sections for call ID: {call_id}")

                if len(question_topics) != num_questions:
                    raise ValueError(f"Mismatch between number of topics and questions for call ID: {call_id}")

                if len(answer_topics) != num_answers:
                    raise ValueError(f"Mismatch between number of topics and answers for call ID: {call_id}")

                # Convert the topics from NumPy array to a list
                if isinstance(section_topics, np.ndarray):
                    section_topics = section_topics.tolist()

                if isinstance(question_topics, np.ndarray):
                    question_topics = question_topics.tolist()

                if isinstance(answer_topics, np.ndarray):
                    answer_topics = answer_topics.tolist()

                # Get the timestamp for the call
                timestamp = value['date']

                # Get company_info
                company_info = value.get('company_info', 'Unknown')

                # Get ceo_participates flag
                ceo_participates = value.get('ceo_participates', False)

                # Get CEO and CFO names
                ceo_names = value.get('ceo_names', [])
                cfo_names = value.get('cfo_names', [])

                # Format sections and topics to be stored correctly in the CSV
                result_dict[call_id] = {
                    "permco": permco,
                    "company_info": company_info,
                    "date": timestamp,
                    "presentation_text": sections,
                    "presentation_topics": section_topics,
                    "participant_questions": questions,
                    "participant_question_topics": question_topics,
                    "management_answers": answers,
                    "management_answer_topics": answer_topics,
                    "ceo_participates": ceo_participates,
                    "ceo_names": ceo_names,
                    "cfo_names": cfo_names
                }
                topic_idx_sections += num_sections
                topic_idx_questions += num_questions
                topic_idx_answers += num_answers

        # Convert results to DataFrame
        records = []
        for call_id, call_data in result_dict.items():
            records.append({
                'permco': call_data['permco'],
                'call_id': call_id,
                'company_info': call_data['company_info'],
                'date': call_data['date'],
                'presentation_text': json.dumps(call_data['presentation_text']),
                'presentation_topics': json.dumps(call_data['presentation_topics']),
                'participant_questions': json.dumps(call_data['participant_questions']),
                'participant_question_topics': json.dumps(call_data['participant_question_topics']),
                'management_answers': json.dumps(call_data['management_answers']),
                'management_answer_topics': json.dumps(call_data['management_answer_topics']),
                'ceo_participates': int(call_data['ceo_participates']),  # Convert bool to int (1/0)
                'ceo_names': json.dumps(call_data['ceo_names']),
                'cfo_names': json.dumps(call_data['cfo_names'])
            })
        # Save the DataFrame
        results_df = pd.DataFrame(records)
        results_output_path = os.path.join(self.index_file_ecc_folder, 'topics_output_zero_300.csv')
        results_df.to_csv(results_output_path, index=False)
        logging.info(f"Results saved to {results_output_path}.")

        # Save the DataFrame for later use (e.g., for topics over time)
        self.results_df = results_df

    def fit_and_save(self, all_relevant_sections, all_relevant_questions, all_management_answers, ecc_sample):
        """
        Fit the BERTopic model, save results, and generate visualizations.
        """
        try:
            total_start_time = time.time()  # Start total time tracking

            # Process presentation sections
            if all_relevant_sections:
                logging.info("Transforming presentation sections with the BERTopic model...")
                transform_start_time = time.time()
                topics_sections, probabilities_sections = self.topic_model.transform(all_relevant_sections)  # Removed verbose=True
                transform_end_time = time.time()
                transform_duration = transform_end_time - transform_start_time
                logging.info(f"Transformed presentation sections in {transform_duration:.2f} seconds.")
            else:
                topics_sections = []
                probabilities_sections = []
                logging.info("No presentation sections to process.")

            # Process participant questions
            if all_relevant_questions:
                logging.info("Transforming participant questions with the BERTopic model...")
                transform_start_time = time.time()
                topics_questions, probabilities_questions = self.topic_model.transform(all_relevant_questions)  # Removed verbose=True
                transform_end_time = time.time()
                transform_duration = transform_end_time - transform_start_time
                logging.info(f"Transformed participant questions in {transform_duration:.2f} seconds.")
            else:
                topics_questions = []
                probabilities_questions = []
                logging.info("No participant questions to process.")

            # Process management answers
            if all_management_answers:
                logging.info("Transforming management answers with the BERTopic model...")
                transform_start_time = time.time()
                topics_answers, probabilities_answers = self.topic_model.transform(all_management_answers)  # Removed verbose=True
                transform_end_time = time.time()
                transform_duration = transform_end_time - transform_start_time
                logging.info(f"Transformed management answers in {transform_duration:.2f} seconds.")
            else:
                topics_answers = []
                probabilities_answers = []
                logging.info("No management answers to process.")

            # Save the results to CSV and store results_df
            logging.info("Saving results...")
            self.save_results(
                all_relevant_sections,
                topics_sections,
                all_relevant_questions,
                topics_questions,
                all_management_answers,
                topics_answers,
                ecc_sample
            )
            logging.info("Results saved.")

            # Generate evaluation files
            logging.info("Generating evaluation files...")
            self.generate_evaluation_files()
            logging.info("Evaluation files generated.")

            # Generate additional visualizations
            self.generate_additional_visualizations()

            # Total execution time
            total_end_time = time.time()
            logging.info(f"Total script execution time: {total_end_time - total_start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"An error occurred in fit_and_save: {e}")
            import traceback
            traceback.print_exc()

    def save_basic_info(self):
        """
        Save basic information about the model.
        """
        start_time = time.time()
        logging.info("Saving basic information...")
        output_file = os.path.join(self.output_dir, "basic_info.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Get the number of topics
            topic_info = self.topic_model.get_topic_info()
            num_topics = len(topic_info)
            f.write(f"Number of Topics: {num_topics}\n\n")

            # Get topic frequency (number of documents per topic)
            f.write("Topics by Frequency:\n")
            f.write(topic_info.to_string())
            f.write("\n\n")

            # Get top words for each topic
            f.write("Top words for each topic:\n")
            for topic_id in topic_info['Topic']:
                if topic_id == -1:
                    f.write("\nTopic -1 (Outliers):\n")
                    f.write("  No words available for outlier topic.\n")
                    continue  # Skip the outlier topic
                top_words = self.topic_model.get_topic(topic_id)
                f.write(f"\nTopic {topic_id}:\n")
                for word, score in top_words:
                    f.write(f"  - {word}: {score}\n")

        end_time = time.time()
        logging.info(f"Basic information saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def save_topics_distribution(self):
        """
        Save a plot of the topic distribution, excluding the outlier topic -1.
        """
        start_time = time.time()
        logging.info("Saving topic distribution...")
        output_file = os.path.join(self.output_dir, "topic_distribution.png")
        topic_info = self.topic_model.get_topic_info()

        # Exclude topic -1
        topic_info = topic_info[topic_info['Topic'] != -1]

        plt.figure(figsize=(10, 6))
        plt.bar(topic_info['Topic'].astype(str), topic_info['Count'])
        plt.xlabel('Topic')
        plt.ylabel('Number of Documents')
        plt.title('Topic Distribution (excluding outlier topic)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        end_time = time.time()
        logging.info(f"Topic distribution saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def save_visualization(self, fig, output_file, file_format="png"):
        """
        Save a visualization from BERTopic's built-in visualizations.

        Args:
            fig: The figure to save.
            output_file: The path to save the figure.
            file_format: The format to save the figure ('png' or 'html').
        """
        start_time = time.time()
        if file_format == "html":
            fig.write_html(output_file)
        else:
            fig.write_image(output_file)
        end_time = time.time()
        logging.info(f"Visualization saved to {output_file} in {end_time - start_time:.2f} seconds.")

    def generate_additional_visualizations(self):
        """
        Generate and save additional visualizations.
        """
        logging.info("Generating additional visualizations...")
        start_time = time.time()

        # Visualize Topics
        logging.info("Visualizing topics...")
        fig = self.topic_model.visualize_topics()
        self.save_visualization(fig, os.path.join(self.output_dir, "topics.html"), file_format="html")

        # Visualize Topic Hierarchy
        logging.info("Visualizing topic hierarchy...")
        fig = self.topic_model.visualize_hierarchy()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_hierarchy.html"), file_format="html")

        # Visualize Topic Terms (BarChart)
        logging.info("Visualizing topic terms...")
        fig = self.topic_model.visualize_barchart()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_barchart.html"), file_format="html")

        # Visualize Topic Similarity (Heatmap)
        logging.info("Visualizing topic similarity...")
        fig = self.topic_model.visualize_heatmap()
        self.save_visualization(fig, os.path.join(self.output_dir, "topic_heatmap.html"), file_format="html")

        # Visualize Term Score Decline (Term Rank)
        logging.info("Visualizing term rank...")
        fig = self.topic_model.visualize_term_rank()
        self.save_visualization(fig, os.path.join(self.output_dir, "term_rank.html"), file_format="html")

        # Visualize Topics over Time
        logging.info("Visualizing topics over time...")
        self.visualize_topics_over_time()

        end_time = time.time()
        logging.info(f"All visualizations generated and saved in {end_time - start_time:.2f} seconds.")

    def visualize_topics_over_time(self):
        """
        Generate and save the Topics over Time visualization.
        """
        try:
            start_time = time.time()

            # Prepare the data
            timestamps = []
            documents = []
            topics_list = []

            for index, row in self.results_df.iterrows():
                date = row['date']  # The date of the conference call

                # For presentation texts
                sections = json.loads(row['presentation_text'])  # List of sections (paragraphs)
                section_topics = json.loads(row['presentation_topics'])  # List of topics for the sections

                num_sections = len(sections)
                timestamps.extend([date] * num_sections)
                documents.extend(sections)
                topics_list.extend(section_topics)

                # For participant questions
                questions = json.loads(row['participant_questions'])
                question_topics = json.loads(row['participant_question_topics'])

                num_questions = len(questions)
                timestamps.extend([date] * num_questions)
                documents.extend(questions)
                topics_list.extend(question_topics)

                # For management answers
                answers = json.loads(row['management_answers'])
                answer_topics = json.loads(row['management_answer_topics'])

                num_answers = len(answers)
                timestamps.extend([date] * num_answers)
                documents.extend(answers)
                topics_list.extend(answer_topics)

            # Convert timestamps to datetime objects
            timestamps = pd.to_datetime(timestamps)

            # Ensure that the number of timestamps matches the number of documents and topics
            if not (len(timestamps) == len(documents) == len(topics_list)):
                raise ValueError("Number of timestamps, documents, and topics do not match.")

            # Set the number of bins to a value lower than 100
            nr_bins = 50  # Adjust as needed

            # Generate topics over time with the specified number of bins
            topics_over_time = self.topic_model.topics_over_time(
                docs=documents,
                topics=topics_list,
                timestamps=timestamps,
                nr_bins=nr_bins
            )

            # Check if topics_over_time is empty
            if topics_over_time.empty:
                logging.warning("No data available for topics over time visualization.")
                return

            # Visualize topics over time with normalize_frequency=True
            fig = self.topic_model.visualize_topics_over_time(
                topics_over_time,
                top_n_topics=10,
                normalize_frequency=True
            )
            self.save_visualization(
                fig,
                os.path.join(self.output_dir, "topics_over_time.html"),
                file_format="html"
            )

            end_time = time.time()
            logging.info(f"Topics over time visualization saved in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"An error occurred in visualize_topics_over_time: {e}")
            import traceback
            traceback.print_exc()

    def generate_evaluation_files(self):
        """
        Generate evaluation files for Presentation, Q&A, and Management Answer sections.
        """
        # Define the sections and their corresponding columns
        sections = [
            {
                'section_type': 'Presentation',
                'text_column': 'presentation_text',
                'topics_column': 'presentation_topics'
            },
            {
                'section_type': 'Participant Questions',
                'text_column': 'participant_questions',
                'topics_column': 'participant_question_topics'
            },
            {
                'section_type': 'Management Answers',
                'text_column': 'management_answers',
                'topics_column': 'management_answer_topics'
            }
        ]

        for section in sections:
            logging.info(f"Generating evaluation file for {section['section_type']} sections...")
            generate_evaluation_file(
                topic_model=self.topic_model,
                results_df=self.results_df,
                output_dir=self.eval_dir,  # Save in 'eval' folder
                text_column=section['text_column'],
                topics_column=section['topics_column'],
                section_type=section['section_type']
            )
            logging.info(f"Evaluation file for {section['section_type']} saved.\n")

def main():
    """
    Main entry point of the script.
    """
    # Start total time tracking
    total_start_time = time.time()

    # Load configuration from config.json
    logging.info("Loading configuration...")
    # Define the relative path to config.json
    relative_path = 'config.json'

    # Define the fallback absolute path
    fallback_path = r'C:\Users\nikla\OneDrive\Dokumente\winfoMaster\Masterarbeit\bertopic_ecc\config.json'

    try:
        # Attempt to load config.json from the relative path
        with open(relative_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from relative path: {os.path.abspath(relative_path)}")
    except FileNotFoundError:
        logging.warning(f"Relative config.json not found at {os.path.abspath(relative_path)}. Trying fallback path...")
        try:
            # Attempt to load config.json from the absolute path
            with open(fallback_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from fallback path: {fallback_path}")
        except FileNotFoundError:
            # Handle the case where both paths fail
            logging.error("Failed to load configuration. config.json not found in both relative and fallback paths.")
            config = {}

    print_configuration(config)

    # Extract variables from the config
    random_seed = config.get("random_seed", 42)  # Provide a default seed
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    index_file_ecc_folder = config.get("index_file_ecc_folder", "index_ecc")
    folderpath_ecc = config.get("folderpath_ecc", "data/ecc")
    sample_size = config.get("sample_size", 100)  # Provide a default sample size
    document_split = config.get("document_split", "paragraph")  # Provide a default method
    max_documents = config.get("max_documents", None)  # Use .get to provide a default value
    model_load_path = config.get("model_load_path", "bertopic_model_dir")

    # Initialize FileHandler and TextProcessor with the imported configuration
    logging.info("Initializing file handler and text processor...")
    # Pass the entire config dictionary to FileHandler
    file_handler = FileHandler(config=config)
    text_processor = TextProcessor(method=document_split)  # Removed 'section_to_analyze' parameter

    # Read the index file
    logging.info("Reading index file...")
    index_file = file_handler.read_index_file()

    # Create ECC sample
    logging.info("Creating ECC sample...")
    ecc_sample = file_handler.create_ecc_sample(sample_size)

    # Extract texts for BERTopic analysis (processed sections/paragraphs)
    logging.info("Extracting and processing relevant sections...")
    extraction_start_time = time.time()  # Time tracking
    all_relevant_sections = []
    all_relevant_questions = []
    all_management_answers = []
    not_considered_count = 0  # Initialize counter
    ecc_sample_filtered = {}  # Create a new dict to hold filtered earnings calls

    for permco, calls in ecc_sample.items():
        calls_filtered = {}
        for call_id, value in calls.items():
            company_info = value.get('company_name', 'Unknown')  # Use .get with default
            date = value.get('date', 'Unknown')
            text = value.get('text_content', '')
            result = text_processor.extract_and_split_section(permco, call_id, company_info, date, text)
            if result and (result.get('presentation_text') or result.get('participant_questions') or result.get('management_answers')):
                # Process presentation_text
                if result.get('presentation_text'):
                    all_relevant_sections.extend(result['presentation_text'])
                # Process participant_questions
                if result.get('participant_questions'):
                    all_relevant_questions.extend(result['participant_questions'])
                # Process management_answers
                if result.get('management_answers'):
                    all_management_answers.extend(result['management_answers'])

                # Add the relevant data to 'value'
                value['presentation_text'] = result['presentation_text'] if result.get('presentation_text') else []
                value['participant_questions'] = result['participant_questions'] if result.get('participant_questions') else []
                value['management_answers'] = result['management_answers'] if result.get('management_answers') else []
                value['participants'] = result.get('participants', [])
                value['ceo_participates'] = result.get('ceo_participates', False)
                value['ceo_names'] = result.get('ceo_names', [])
                value['cfo_names'] = result.get('cfo_names', [])
                # Add 'company_info' to 'value'
                value['company_info'] = company_info
                # Add the call to calls_filtered
                calls_filtered[call_id] = value
            else:
                logging.warning(f"Earnings call {call_id} has no relevant sections and will be excluded.")
                not_considered_count += 1
        if calls_filtered:
            ecc_sample_filtered[permco] = calls_filtered
    extraction_end_time = time.time()
    logging.info(f"Extraction and processing completed in {extraction_end_time - extraction_start_time:.2f} seconds.")
    logging.info(f"Total number of earnings calls not considered due to missing sections: {not_considered_count}")

    if not all_relevant_sections and not all_relevant_questions and not all_management_answers:
        logging.warning("No relevant sections or participant questions or management answers found to fit BERTopic.")
        return

    # Instantiate BertopicFitting and process the data
    bertopic_fitting = BertopicFitting(config, model_load_path)
    bertopic_fitting.fit_and_save(all_relevant_sections, all_relevant_questions, all_management_answers, ecc_sample_filtered)

    # Total execution time
    total_end_time = time.time()
    logging.info(f"Total script execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
