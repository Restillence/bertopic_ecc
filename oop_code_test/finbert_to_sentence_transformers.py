import os
import json
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models

# Step 1: Load the configuration file where the model path is defined
config_path = 'C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json'  # Update with the actual path to your config.json

with open(config_path, 'r') as f:
    config = json.load(f)

# Step 2: Get the model path from the config
model_path = config.get("finbert_model_path")

if model_path is None:
    raise ValueError("Model path not found in the configuration file.")

# Step 3: Load the BERT model and tokenizer from the Hugging Face hub
model_name = 'yiyanghkust/finbert-pretrain'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Step 4: Create a SentenceTransformer model from the BERT model
word_embedding_model = models.Transformer(model_name_or_path=model_name)

# Add a pooling layer to the model
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,  # Use mean pooling
    pooling_mode_cls_token=False,   # Do not use CLS token pooling
    pooling_mode_max_tokens=False   # Do not use max token pooling
)

# Combine the transformer and pooling model into a SentenceTransformer model
sentence_transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Step 5: Save the SentenceTransformer model to the specified path in the config
sentence_transformer_model.save(model_path)
