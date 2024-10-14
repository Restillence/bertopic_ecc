#text_processing.py
#Allows to split the text into sentences or paragraphs, depending on the method specified.
import re
from nltk.tokenize import sent_tokenize

class TextProcessor:
    def __init__(self, method, section_to_analyze):
        """
        Initialize a TextProcessor object.

        Parameters
        ----------
        method : str
            The method to use for text splitting. Options are 'sentences', 'paragraphs', or 'custom'.
        section_to_analyze : str
            The section to analyze, either 'Presentation' or 'Questions and Answers'.
        """
        self.method = method
        self.section_to_analyze = section_to_analyze

    def remove_unwanted_sections(self, text):
        """
        Remove blocks like "TEXT version of Transcript", "Corporate Participants", "Conference Call Participants" from a transcript text.

        Parameters
        ----------
        text : str
            The text from which to remove the unwanted sections.

        Returns
        -------
        str
            The text with the unwanted sections removed.
        """
        pattern = r'(TEXT version of Transcript|Corporate Participants|Conference Call Participants)\n=+\n(?:.*\n)*?=+\n'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return cleaned_text

    def remove_questions_and_answers_and_beyond(self, text):
        """
        Remove the "Questions and Answers" section and everything after it.

        Parameters
        ----------
        text : str
            The text from which to remove the "Questions and Answers" section.

        Returns
        -------
        str
            The text with the "Questions and Answers" section removed.
        """
        if self.section_to_analyze.lower() != "questions and answers":
            pattern = r'Questions and Answers\s*\n[-=]+\n.*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def remove_concluding_statements(self, text):
        """
        Remove concluding statements that might be present after the main content.

        Parameters
        ----------
        text : str
            The text from which to remove the concluding statements.

        Returns
        -------
        str
            The text with the concluding statements removed.
        """
        pattern = r'(Ladies and gentlemen, thank you for participating.*|This concludes today\'s program.*|You may all disconnect.*)'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def remove_pattern(self, text):
        """
        Remove patterns like "---- some text ----"

        Parameters
        ----------
        text : str
            The text from which to remove the patterns.

        Returns
        -------
        str
            The text with the patterns removed.
        """
        pattern = r'-{4,}.*?-{4,}'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text

    def remove_specific_string(self, text):
        """
        Forcefully remove "Presentation" from the text, regardless of where it appears.
        
        Parameters
        ----------
        text : str
            The text from which to remove specific strings.

        Returns
        -------
        str
            The text with the specific strings removed.
        """
        # Remove "TEXT version of Transcript" case-insensitively
        text = re.sub(r"TEXT version of Transcript", '', text, flags=re.IGNORECASE)
        
        # Forcefully remove "Presentation" anywhere it appears at the start of a line, along with optional whitespace
        text = re.sub(r'(?i)^\s*Presentation\s*\n', '', text, flags=re.MULTILINE)
        
        return text.strip()

    def remove_separator_line(self, text):
        """
        Remove lines that are just a series of "=" or "-" characters

        Parameters
        ----------
        text : str
            The text from which to remove the separator lines.

        Returns
        -------
        str
            The text with the separator lines removed.
        """
        return re.sub(r'^\s*[-=]{3,}\s*$', '', text, flags=re.MULTILINE).strip()

    def split_text(self, text):
        """
        Split text into sentences or paragraphs, depending on the method specified.

        Parameters
        ----------
        text : str
            The text to be split.

        Returns
        -------
        list of str
            A list of sentences or paragraphs, depending on the method.

        Raises
        ------
        ValueError
            If the method is not 'sentences', 'paragraphs', or 'custom'.
        """
        if self.method == 'sentences':
            return sent_tokenize(text)
        elif self.method == 'paragraphs':
            paragraphs = re.split(r'\n{2,}', text)
            return [para.strip() for para in paragraphs if para.strip()]
        elif self.method == 'custom':
            return re.split(r'\.\s\s|\n\n', text)
        else:
            raise ValueError("Invalid text splitting method. Choose 'sentences', 'paragraphs', or 'custom'.")

    def remove_presentation_from_final_list(self, text_list):
        """
        Ensure the string 'Presentation' is fully removed from the final list of sentences or paragraphs.
        
        Parameters
        ----------
        text_list : list of str
            The list of sentences or paragraphs.

        Returns
        -------
        list of str
            The cleaned list with 'Presentation' removed.
        """
        return [element for element in text_list if element.strip().lower() != "presentation"]

    def filter_short_elements(self, text_list):
        """
        Remove elements that have fewer than 3 words from the list.
        
        Parameters
        ----------
        text_list : list of str
            The list of sentences or paragraphs.

        Returns
        -------
        list of str
            The filtered list where each element has at least 3 words.
        """
        return [element for element in text_list if len(element.split()) >= 3]

    def extract_and_split_section(self, company, call_id, company_info, date, text):
        """
        Extract and split the relevant section from the text.

        Parameters
        ----------
        company : str
            The company name.
        call_id : str
            The call ID.
        company_info : dict
            A dictionary containing information about the company.
        date : str
            The date of the call.
        text : str
            The text to be split.

        Returns
        -------
        list of str
            A list of sentences or paragraphs, depending on the method.

        Raises
        ------
        ValueError
            If the method is not 'sentences', 'paragraphs', or 'custom'.
        """
        # Proceed with cleaning the text before splitting
        text = self.remove_unwanted_sections(text)
        text = self.remove_questions_and_answers_and_beyond(text)
        text = self.remove_concluding_statements(text)
        text = self.remove_pattern(text)
        text = self.remove_specific_string(text)  # Ensure "Presentation" is removed forcefully

        # Split the cleaned text
        if self.method == 'sentences':
            combined_text = self.split_text(text)
        else:
            combined_text = self.split_text_by_visual_cues(text)

        # Final cleanup: Remove any remaining separator lines
        if isinstance(combined_text, list):
            combined_text = [self.remove_separator_line(para) for para in combined_text]
        else:
            combined_text = self.remove_separator_line(combined_text)

        # Final Steps: Remove "Presentation" and filter out elements with fewer than 3 words
        combined_text = self.remove_presentation_from_final_list(combined_text)
        combined_text = self.filter_short_elements(combined_text)

        return combined_text

    def split_text_by_visual_cues(self, text):
        """
        Split text into paragraphs by visual cues (e.g. blank lines, separators, etc.) and remove any unnecessary paragraphs.

        Parameters
        ----------
        text : str
            The text to be split.

        Returns
        -------
        list of str
            A list of paragraphs, with any unnecessary ones removed.
        """
        pattern = r'\n\s*\n|\n[=]+\n|\n[-]+\n|\n\s{2,}\n|(?:^|\n)(?=[A-Z][a-z]+, [a-zA-Z\s]*[-]*[0-9]*)|\.\s*\n'
        paragraphs = re.split(pattern, text)
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        paragraphs = self.preprocess_paragraphs(paragraphs)
        return paragraphs

    def preprocess_paragraphs(self, paragraphs):
        """
        Remove unnecessary paragraphs from the list of paragraphs.

        Parameters
        ----------
        paragraphs : list of str
            The list of paragraphs to be cleaned.

        Returns
        -------
        list of str
            The cleaned list of paragraphs.
        """
        cleaned_paragraphs = []
        skip_next = False

        for i, paragraph in enumerate(paragraphs):
            if skip_next:
                skip_next = False
                continue

            # Skip separator lines followed by name/title paragraphs
            if self.is_separator_line(paragraph) and (i + 1 < len(paragraphs)):
                if self.is_name_title_paragraph(paragraphs[i + 1]):
                    skip_next = True
                    continue

            # Remove name/title paragraphs and separator lines
            if not self.is_name_title_paragraph(paragraph) and not self.is_separator_line(paragraph):
                if self.has_content_indicators(paragraph) or len(paragraph.split()) > 2:
                    cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def is_name_title_paragraph(self, paragraph):
        """
        Check if a paragraph is a name/title paragraph.

        Parameters
        ----------
        paragraph : str
            The paragraph to be checked.

        Returns
        -------
        bool
            True if the paragraph is a name/title paragraph, False otherwise.
        """
        pattern = r"^[A-Z][a-zA-Z\. ]+, [A-Z][a-zA-Z\. ]+, [A-Za-z\.,&; ]+ - [A-Z]{2,}.*$"
        return re.match(pattern, paragraph.strip()) is not None

    def has_content_indicators(self, paragraph):
        """
        Check if a paragraph contains any of the following phrases which are indicators of a presentation section.

        Parameters
        ----------
        paragraph : str
            The paragraph to be checked.

        Returns
        -------
        bool
            True if the paragraph contains any of the phrases, False otherwise.
        """
        indicators = [
            "pleasure to", "turn the call over", "let me introduce", "i will now hand over",
            "welcome", "thank", "let's get started"
        ]
        return any(phrase in paragraph.lower() for phrase in indicators)

    def is_separator_line(self, paragraph):
        """
        Check if a paragraph is a separator line.

        Parameters
        ----------
        paragraph : str
            The paragraph to be checked.

        Returns
        -------
        bool
            True if the paragraph is a separator line, False otherwise.
        """
        pattern = r"^[-=]{3,}$"
        return re.match(pattern, paragraph.strip()) is not None

    def extract_all_relevant_sections(self, ecc_sample, max_documents):
        """
        Extract all relevant sections from the ECC sample.

        Parameters
        ----------
        ecc_sample : dict
            The ECC sample to extract relevant sections from.
        max_documents : int
            The maximum number of documents to extract relevant sections from.

        Returns
        -------
        list
            A list of all relevant sections extracted from the ECC sample.
        """
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
