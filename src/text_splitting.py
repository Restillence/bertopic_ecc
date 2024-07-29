import re
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_text(text, method):
    print("Splitting text using method:", method)
    if method == 'sentences':
        return sent_tokenize(text)
    elif method == 'paragraphs': #currently this splits each ecc into a list of 3 items. 1st and 2nd item are
        #participant info, 3rd item is the text
        paragraphs = re.split(r'\n{2,}', text)
        return [para.strip() for para in paragraphs if para.strip()]
    elif method == 'custom':
        # Define your custom splitting method here
        # Example: Split by double newlines or periods followed by two spaces
        return re.split(r'\.\s\s|\n\n', text)
    else:
        raise ValueError("Invalid text splitting method. Choose 'sentences', 'paragraphs', or 'custom'.")

def remove_unnecessary_sections(text):
