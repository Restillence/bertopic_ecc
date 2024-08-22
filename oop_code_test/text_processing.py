import re
from nltk.tokenize import sent_tokenize

class TextProcessor:
    def __init__(self, method, section_to_analyze):
        self.method = method
        self.section_to_analyze = section_to_analyze

    def split_text(self, text):
        #print("Splitting text using method:", self.method)#debugging line
        if self.method == 'sentences':
            return sent_tokenize(text)
        elif self.method == 'paragraphs':
            paragraphs = re.split(r'\n{2,}', text)
            return [para.strip() for para in paragraphs if para.strip()]
        elif self.method == 'custom':
            return re.split(r'\.\s\s|\n\n', text)
        else:
            raise ValueError("Invalid text splitting method. Choose 'sentences', 'paragraphs', or 'custom'.")

    def extract_and_split_section(self, company, call_id, company_info, date, text):
        #print(f"Processing text for company: {company_info} with permco: {company}, call ID: {call_id}") # debugging line
        paragraphs = self.split_text(text)

        section_patterns = {
            "Presentation": r"Presentation\s*\n[-=]+",
            "Questions and Answers": r"Questions and Answers\s*\n[-=]+"
        }

        relevant_section = None
        start_index = None
        for i, element in enumerate(paragraphs):
            if re.search(section_patterns.get(self.section_to_analyze, ""), element, re.IGNORECASE):
                relevant_section = element
                start_index = i
                break

        if relevant_section is None:
            print(f"No explicit section found for {self.section_to_analyze} in company: {company}, call ID: {call_id}. Using the first part as the Presentation section.")
            relevant_section = paragraphs[0]  # Assume the first part is the presentation
            start_index = 0

        combined_text = '\n\n'.join(paragraphs[start_index:])

        if self.section_to_analyze == "Presentation":
            qa_match = re.search(section_patterns["Questions and Answers"], combined_text, re.IGNORECASE)
            if qa_match:
                combined_text = combined_text[:qa_match.start()]
        elif self.section_to_analyze == "Questions and Answers":
            qa_match = re.search(section_patterns["Questions and Answers"], combined_text, re.IGNORECASE)
            if qa_match:
                combined_text = combined_text[qa_match.start():]

        if self.method == 'sentences':
            return self.split_text(combined_text)
        
        return self.split_text_by_visual_cues(combined_text)

    def split_text_by_visual_cues(self, text):
        pattern = r'\n\s*\n|\n[=]+\n|\n[-]+\n|\n\s{2,}\n|(?:^|\n)(?=[A-Z][a-z]+, [a-zA-Z\s]*[-]*[0-9]*)|\.\s*\n'
        paragraphs = re.split(pattern, text)
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs

    def extract_all_relevant_sections(self, ecc_sample, max_documents):
        all_relevant_sections = []
        document_count = 0

        for permco, calls in ecc_sample.items():
            for call_id, value in calls.items():
                if document_count >= max_documents:
                    break
                company_info, date, text = value
                relevant_sections = self.extract_and_split_section(permco, call_id, company_info, date, text)
                if relevant_sections is not None:
                    if isinstance(relevant_sections, list):
                        all_relevant_sections.extend(relevant_sections)
                    else:
                        all_relevant_sections.append(relevant_sections)
                    document_count += 1
            if document_count >= max_documents:
                break
        print(f"Extracted {len(all_relevant_sections)} relevant sections.")
        return all_relevant_sections
