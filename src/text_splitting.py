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

def extract_and_split_section(company, call_id, company_info, date, text, document_split, section_to_analyze):
    print(f"Processing text for company: {company}, call ID: {call_id}")
    # First, split the text into paragraphs to locate the section
    paragraphs = split_text(text, 'paragraphs')
    
    # Define patterns for the sections
    section_patterns = {
        "Presentation": r"Presentation\s*\n[-=]+",
        "Questions and Answers": r"Questions and Answers\s*\n[-=]+"
    }

    # Find the relevant section
    relevant_section = None
    start_index = None
    for i, element in enumerate(paragraphs):
        print(f"Paragraph {i}: {element[:100]}...")  # Print first 100 characters of each paragraph for debugging
        if re.search(section_patterns[section_to_analyze], element, re.IGNORECASE):
            relevant_section = element
            start_index = i
            break
    
    if relevant_section is None:
        print(f"No relevant section found for {section_to_analyze} in company: {company}, call ID: {call_id}")
        return None  # Return None if the specified section is not found

    # Combine paragraphs starting from the relevant section
    combined_text = '\n\n'.join(paragraphs[start_index:])
    
    # Split out the next section if it's "Questions and Answers" or remove text before it if analyzing "Questions and Answers"
    if section_to_analyze == "Presentation":
        qa_match = re.search(r"Questions and Answers\s*\n[-=]+", combined_text, re.IGNORECASE)
        if qa_match:
            combined_text = combined_text[:qa_match.start()]
    elif section_to_analyze == "Questions and Answers":
        qa_match = re.search(section_patterns["Questions and Answers"], combined_text, re.IGNORECASE)
        if qa_match:
            combined_text = combined_text[qa_match.start():]

    # If the split method is sentences, split and return the sentences directly
    if document_split == 'sentences':
        return split_text(combined_text, 'sentences')
    
    # Otherwise, further split the relevant section using visual cues
    return split_text_by_visual_cues(combined_text)

def split_text_by_visual_cues(text):
    # Define a pattern to split based on multiple newlines, lines containing only "=", "-", or more spaces followed by a new line,
    # as well as periods followed by a newline or a space and a newline
    pattern = r'\n\s*\n|\n[=]+\n|\n[-]+\n|\n\s{2,}\n|(?:^|\n)(?=[A-Z][a-z]+, [a-zA-Z\s]*[-]*[0-9]*)|\.\s*\n'
    
    # Split the text based on the pattern
    paragraphs = re.split(pattern, text)
    
    # Clean up and return non-empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    return paragraphs
