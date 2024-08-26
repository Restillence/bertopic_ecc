# %%
from bertopic import BERTopic
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
import json
from sentence_transformers import SentenceTransformer
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
import os 
from utils import print_configuration

# %%
# Load configuration from config.json  
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

# Initialize FileHandler and TextProcessor with the imported configuration
file_handler = FileHandler(index_file_path=index_file_path, folderpath_ecc=folderpath_ecc)
text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

# %%
# Create the sample and extract relevant sections
index_file = file_handler.read_index_file()
ecc_sample = file_handler.create_ecc_sample(sample_size)
all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)
print(all_relevant_sections[:500])


# %%
#Initialize model
model_path = config.get("finbert_model_path")
if model_path is None:
    raise ValueError("Model path not found in the configuration file.")

# Ensure that the model path is correct
if not os.path.exists(model_path):
    raise ValueError(f"The specified model path does not exist: {model_path}")

model = SentenceTransformer(model_path)
topic_model = BERTopic(embedding_model=model)

# Fit the BERTopic model
print("Fitting BERTopic...")
topics, probs = topic_model.fit_transform(all_relevant_sections)

