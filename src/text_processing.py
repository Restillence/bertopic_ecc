# text_processing.py
# Allows splitting the text into sentences or paragraphs, depending on the method specified.

import re
import json
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

        # Define title variations for CEO and CFO
        self.ceo_titles = [
            r'\bCEO\b',
            r'\bChief Executive Officer\b',
            r'\bChief Exec Officer\b',
            r'\bPresident and CEO\b',
            r'\bChairman and CEO\b',
            r'\bChief Executive Officer/Chief Financial Officer\b',
            r'\bChief Executive Officer\s*/\s*Chief Financial Officer\b',
            r'\bChief Operating Officer and CEO\b',
            r'\bCEO and President\b'
        ]

        self.cfo_titles = [
            r'\bCFO\b',
            r'\bChief Financial Officer\b',
            r'\bFinance Director\b',
            r'\bVice President of Finance\b',
            r'\bSenior Vice President and CFO\b',
            r'\bChief Financial Officer\s*/\s*Chief Executive Officer\b',
            r'\bChief Financial Officer and Chief Operating Officer\b'
        ]

    def extract_participants(self, text):
        """
        Extract the list of corporate participants and their positions from the transcript text.

        Parameters
        ----------
        text : str
            The transcript text.

        Returns
        -------
        participants : list of dict
            A list of dictionaries with keys 'name' and 'position'.
        """
        participants = []
        # Search for the 'Corporate Participants' section
        pattern = r'(?:\n|^)(Corporate Participants|Company Participants|Participants)\n=+\n(.*?)(?=\n=+\n|\Z)'
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            participants_text = match.group(2)
            # Split participants by entries starting with '*'
            participant_entries = re.split(r'\n\s*\*\s*', participants_text)
            participant_entries = [entry.strip() for entry in participant_entries if entry.strip()]
            for entry in participant_entries:
                lines = entry.strip().split('\n')
                if len(lines) >= 2:
                    name = lines[0].strip()
                    position_line = lines[1].strip()
                    # Split position_line on ' - ' to separate company and position
                    parts = re.split(r'\s*[-–:]\s*', position_line, maxsplit=1)
                    if len(parts) == 2:
                        # company_name = parts[0]  # Not used here
                        position = parts[1]
                    else:
                        position = position_line
                else:
                    # Handle cases where position is on the same line as name
                    entry_text = ' '.join(lines)
                    name_position = re.split(r'\s*[-–:]\s*', entry_text, maxsplit=1)
                    name = name_position[0].strip()
                    position = name_position[1].strip() if len(name_position) > 1 else ''

                # **New code to clean and standardize the name**
                name = self.clean_and_standardize_name(name)
                participants.append({'name': name, 'position': position})
        else:
            print("Warning: 'Corporate Participants' section not found in the transcript for participant extraction.")
        return participants

    def clean_and_standardize_name(self, name):
        """
        Clean and standardize the participant's name to 'F. Lastname' format.

        Parameters
        ----------
        name : str
            The raw name string.

        Returns
        -------
        str
            The cleaned and standardized name.
        """
        # Remove any leading/trailing non-alphabetic characters and whitespace
        name = re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', name).strip()

        # Remove suffixes like Jr., Sr., III, etc.
        suffixes = ['Jr.', 'Sr.', 'Jr', 'Sr', 'II', 'III', 'IV', 'V']
        name_parts = name.split()
        name_parts = [part for part in name_parts if part not in suffixes]

        if len(name_parts) == 0:
            return name  # If all parts are suffixes, return the original name

        # Get the first word and the last word
        first_word = name_parts[0]
        last_word = name_parts[-1]

        # Extract the first initial using regex to handle cases like "R." or "R"
        first_initial_match = re.match(r'^([A-Za-z])\.', first_word)
        if first_initial_match:
            first_initial = first_initial_match.group(1).upper() + '.'
        else:
            first_initial = first_word[0].upper() + '.'

        # Combine as 'F. Lastname'
        standardized_name = f"{first_initial} {last_word}"

        return standardized_name.strip()

    def extract_analysts(self, text):
        """
        Extract the list of analyst participants from the transcript text.

        Parameters
        ----------
        text : str
            The transcript text.

        Returns
        -------
        analysts : list of dict
            A list of dictionaries with keys 'name' and 'firm'.
        """
        analysts = []
        # Search for the 'Conference Call Participants' section
        pattern = r'(?:\n|^)(Conference Call Participants|Analysts)\n=+\n(.*?)(?=\n=+\n|\Z)'
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            participants_text = match.group(2)
            # Split participants by entries starting with '*'
            participant_entries = re.split(r'\n\s*\*\s*', participants_text)
            participant_entries = [entry.strip() for entry in participant_entries if entry.strip()]
            for entry in participant_entries:
                lines = entry.strip().split('\n')
                if len(lines) >= 1:
                    name_firm = lines[0].strip()
                    # Split name and firm using ' - ' or '–' or ':' or similar delimiters
                    parts = re.split(r'\s*[-–:]\s*', name_firm, maxsplit=1)
                    name = self.clean_and_standardize_name(parts[0].strip())
                    firm = parts[1].strip() if len(parts) > 1 else ''
                    analysts.append({'name': name, 'firm': firm})
        else:
            print("Warning: 'Conference Call Participants' section not found in the transcript for analyst extraction.")
        return analysts

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
        pattern = r'(TEXT version of Transcript|Corporate Participants|Company Participants|Participants|Conference Call Participants)\n=+\n(?:.*?\n)*?(?=\n=+\n|\Z)'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_text

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
        dict
            A dictionary containing 'combined_text', 'participants', 'ceo_participates', 'ceo_names', and 'cfo_names'.
        """
        # First, extract participants from the original text
        participants = self.extract_participants(text)

        # Determine if CEO participates
        ceo_participates = any(self.is_ceo(p['position']) for p in participants)

        # Store CEO and CFO names
        ceo_names = [p['name'] for p in participants if self.is_ceo(p['position'])]
        cfo_names = [p['name'] for p in participants if self.is_cfo(p['position'])]

        # Proceed with the extraction based on the selected section
        if self.section_to_analyze.lower() == "questions and answers":
            # Adjusted pattern to match 'Questions and Answers' as a section heading
            pattern = r'^\s*(Questions?\s+and\s+Answers?)\s*$'
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                # Take everything after the 'Questions and Answers' heading
                start_index = match.end()
                text = text[start_index:]
                #print(f"'Questions and Answers' section extracted for call ID: {call_id}")
            else:
                print(f"'Questions and Answers' section not found for call ID: {call_id}")
                text = ''
        elif self.section_to_analyze.lower() == "presentation":
            # Adjusted pattern to match 'Presentation' as a section heading
            start_pattern = r'^\s*Presentation\s*$'
            end_pattern = r'^\s*(Questions?\s+and\s+Answers?)\s*$'

            start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            end_match = re.search(end_pattern, text, flags=re.IGNORECASE | re.MULTILINE)

            if start_match:
                start_index = start_match.end()
                if end_match:
                    end_index = end_match.start()
                    text = text[start_index:end_index]
                    #print(f"'Presentation' section extracted for call ID: {call_id}")
                else:
                    text = text[start_index:]
                    print(f"'Presentation' section extracted up to end of text for call ID: {call_id}")
            else:
                print(f"'Presentation' section not found for call ID: {call_id}")
                # Attempt to extract from start up to 'Questions and Answers' section
                if end_match:
                    end_index = end_match.start()
                    text = text[:end_index]
                    print(f"Extracted text up to 'Questions and Answers' for call ID: {call_id}")
                else:
                    print(f"'Questions and Answers' section not found for call ID: {call_id}")
                    print(f"No 'Presentation' or 'Questions and Answers' section found for call ID: {call_id}")
                    text = ''  # Set text to empty string

        if not text.strip():
            print(f"No relevant sections found for call ID: {call_id}")
            return None

        # Proceed with cleaning the text after extracting the section
        text = self.remove_unwanted_sections(text)
        text = self.remove_concluding_statements(text)
        text = self.remove_pattern(text)
        text = self.remove_specific_string(text)  # Ensure "Presentation" is removed forcefully

        # Split the cleaned text
        if self.method == 'sentences':
            combined_text = self.split_text(text)
        else:
            combined_text = self.split_text_by_visual_cues(text)

        # Final cleanup: Remove any remaining separator lines
        combined_text = [self.remove_separator_line(para) for para in combined_text]
        # Remove any empty strings resulted from cleaning
        combined_text = [para for para in combined_text if para.strip()]
        # Final Steps: Remove "Presentation" and filter out elements with fewer than 3 words
        combined_text = self.remove_presentation_from_final_list(combined_text)
        combined_text = self.filter_short_elements(combined_text)

        result = {
            'combined_text': combined_text,
            'participants': participants,
            'ceo_participates': int(ceo_participates),  # Ensure it's an integer (0 or 1)
            'ceo_names': ceo_names,
            'cfo_names': cfo_names
        }

        return result

    def split_text_by_visual_cues(self, text):
        """
        Split text into paragraphs by visual cues (e.g., blank lines, separators, etc.) and remove any unnecessary paragraphs.

        Parameters
        ----------
        text : str
            The text to be split.

        Returns
        -------
        list of str
            The list of paragraphs, with any unnecessary ones removed.
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

    def is_ceo(self, position):
        """
        Determine if the position corresponds to a CEO.

        Parameters
        ----------
        position : str
            The position title.

        Returns
        -------
        bool
            True if the position is a CEO, False otherwise.
        """
        for title_pattern in self.ceo_titles:
            if re.search(title_pattern, position, flags=re.IGNORECASE):
                return True
        return False

    def is_cfo(self, position):
        """
        Determine if the position corresponds to a CFO.

        Parameters
        ----------
        position : str
            The position title.

        Returns
        -------
        bool
            True if the position is a CFO, False otherwise.
        """
        for title_pattern in self.cfo_titles:
            if re.search(title_pattern, position, flags=re.IGNORECASE):
                return True
        return False

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
                company_info = value['company_name']
                date = value['date']
                text = value['text_content']
                result = self.extract_and_split_section(permco, call_id, company_info, date, text)
                if result and result['combined_text']:
                    all_relevant_sections.extend(result['combined_text'])
                    # Add the relevant data to 'value'
                    value['relevant_sections'] = result['combined_text']
                    value['participants'] = result['participants']
                    value['ceo_participates'] = result['ceo_participates']
                    value['ceo_names'] = result['ceo_names']
                    value['cfo_names'] = result['cfo_names']
                    document_count += 1
                else:
                    print(f"No relevant sections found for call ID: {call_id}")
            if document_count >= max_documents:
                break

        return all_relevant_sections
