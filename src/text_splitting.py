import re
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_text(text, method):
    print("Splitting text using method:", method)
    if method == 'sentences':
        return sent_tokenize(text)
    elif method == 'paragraphs':
        return text.split('\n\n')
    elif method == 'custom':
        # Define your custom splitting method here
        # Example: Split by double newlines or periods followed by two spaces
        return re.split(r'\.\s\s|\n\n', text)
    else:
        raise ValueError("Invalid text splitting method. Choose 'sentences', 'paragraphs', or 'custom'.")
