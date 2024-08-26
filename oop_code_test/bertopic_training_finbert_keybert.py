import os
import json
import time
from bertopic import BERTopic
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT  # Import KeyBERT

# Load configuration from config.json  
print("Loading configuration...")
with open('C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract variables from the config
index_file_ecc_folder = config["index_file_ecc_folder"]
folderpath_ecc = config["folderpath_ecc"]
model_save_path = config["model_save_path"]
index_file_path = config["index_file_path"]
embeddings_path = config["embeddings_path"]
finbert_model_path = config["finbert_model_path"]

sample_size = config["sample_size"]
document_split = config["document_split"]
random_seed = config["random_seed"]
section_to_analyze = config["section_to_analyze"]
max_documents = config["max_documents"]

keybert_params = config["keybert_params"]

# Initialize FileHandler and TextProcessor with the imported configuration
print("Initializing file handler and text processor...")
file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

# Create the sample and extract relevant sections
print("Reading index file and creating ECC sample...")
index_file = file_handler.read_index_file()
ecc_sample = file_handler.create_ecc_sample(sample_size)
all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)
print(f"Extracted {len(all_relevant_sections)} relevant sections for analysis.")

# Initialize the SentenceTransformer model
print("Loading SentenceTransformer model...")
model_path = config.get("finbert_model_path")
if model_path is None:
    raise ValueError("Model path not found in the configuration file.")

# Ensure that the model path is correct
if not os.path.exists(model_path):
    raise ValueError(f"The specified model path does not exist: {model_path}")

model = SentenceTransformer(model_path)

# Initialize BERTopic with the custom SentenceTransformer model
print("Initializing BERTopic model...")
topic_model = BERTopic(embedding_model=model)

# Check if there are relevant sections to fit
if all_relevant_sections:
    print("Training BERTopic model...")
    start_time = time.time()

    # Fit the BERTopic model
    topics, probs = topic_model.fit_transform(all_relevant_sections)

    end_time = time.time()

    # Print information about the training process
    print(f"BERTopic model trained on {len(all_relevant_sections)} sections.")
    print(f"Number of topics generated: {len(set(topics))}")
    print(f"Training time: {end_time - start_time:.2f} seconds.")

    # Save the BERTopic model to the specified path
    print("Saving BERTopic model...")
    topic_model.save(model_save_path)
    print(f"BERTopic model saved to {model_save_path}.")
else:
    print("No relevant sections found to fit BERTopic.")

# Initialize KeyBERT with the same SentenceTransformer model
print("Initializing KeyBERT for keyword extraction...")
kw_model = KeyBERT(model=model)

# Extract keywords for each topic using KeyBERT, with parameters from config.json
print("Extracting keywords for each topic...")
for topic in range(len(topic_model.get_topics())):
    topic_words = topic_model.get_topic(topic)
    if topic_words:
        topic_keywords = kw_model.extract_keywords(
            " ".join([word[0] for word in topic_words]), 
            keyphrase_ngram_range=tuple(keybert_params["keyphrase_ngram_range"]),
            stop_words=keybert_params["stop_words"],
            use_maxsum=keybert_params["use_maxsum"],
            use_mmr=keybert_params["use_mmr"],
            diversity=keybert_params["diversity"],
            top_n=keybert_params["top_n"]
        )
        print(f"Topic {topic}: {topic_keywords}")
    else:
        print(f"Topic {topic} has no significant words.")
