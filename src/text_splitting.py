import re
from nltk.tokenize import sent_tokenize

def split_text(text, method):
    print("Splitting text using method:", method)
    if method == 'sentences':
        return sent_tokenize(text)
    elif method == 'paragraphs':
        # Split based on a specific pattern that identifies paragraphs, e.g., double newlines
        paragraphs = re.split(r'\n{2,}', text)
        return [para.strip() for para in paragraphs if para.strip()]
    elif method == 'custom':
        # Define your custom splitting method here
        # Example: Split by double newlines or periods followed by two spaces
        return re.split(r'\.\s\s|\n\n', text)
    else:
        raise ValueError("Invalid text splitting method. Choose 'sentences', 'paragraphs', or 'custom'.")

def process_texts(company, call_id, company_info, date, text, document_split):
    #right now extracts the "Presentation" section
    print(f"Splitting text for company: {company}, call ID: {call_id}")
    split_texts = split_text(text, document_split)
    
    # Find the first element containing the word "Presentation"
    for i, element in enumerate(split_texts):
        if "Presentation" in element:
            return element
    
    return None  # Return None if "Presentation" is not found in any element



def split_text_by_visual_cues(text):
    #complex splitting pattern
    # Define a pattern to split based on multiple newlines, lines containing only "=", "-", or more spaces followed by a new line,
    # as well as periods followed by a newline or a space and a newline
    pattern = r'\n\s*\n|\n[=]+\n|\n[-]+\n|\n\s{2,}\n|(?:^|\n)(?=[A-Z][a-z]+, [a-zA-Z\s]*[-]*[0-9]*)|\.\s*\n'

    
    # Split the text based on the pattern
    paragraphs = re.split(pattern, text)
    
    # Clean up and return non-empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    return paragraphs






