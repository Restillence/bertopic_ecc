from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models

# Step 1: Load the BERT model and tokenizer from the Hugging Face hub
model_name = 'yiyanghkust/finbert-pretrain'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Step 2: Create a SentenceTransformer model from the BERT model
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

# Step 3: Save the SentenceTransformer model
sentence_transformer_model.save('finbert-sentence-transformer')

"""
# Step 4: Load the SentenceTransformer model (optional, for verification)
model = SentenceTransformer('finbert-sentence-transformer')

# Example sentences
sentences = ["The market is going up.", "The financial report was released."]

# Step 5: Encode sentences to get sentence embeddings
embeddings = model.encode(sentences)

# Print the embeddings
print(embeddings)
"""