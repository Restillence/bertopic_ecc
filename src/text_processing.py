# text_processing.py

import re
from nltk.tokenize import sent_tokenize

class TextProcessor:
    def __init__(self, method):
        self.method = method

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
        """
        participants = []
        # Search for the 'Corporate Participants' section
        pattern = r'(?:\n|^)(Corporate Participants|Company Participants)\n=+\n(.*?)(?=\n=+\n|\Z)'
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
                        position = parts[1]
                    else:
                        position = position_line
                else:
                    # Handle cases where position is on the same line as name
                    entry_text = ' '.join(lines)
                    name_position = re.split(r'\s*[-–:]\s*', entry_text, maxsplit=1)
                    name = name_position[0].strip()
                    position = name_position[1].strip() if len(name_position) > 1 else ''

                # Clean and standardize the name
                name = self.clean_and_standardize_name(name)
                participants.append({'name': name, 'position': position})
        else:
            print("Warning: 'Corporate Participants' section not found in the transcript for participant extraction.")
        return participants

    def extract_analysts(self, text):
        """
        Extract the list of analyst participants from the transcript text.
        """
        analysts = []
        # Search for the 'Conference Call Participants' or 'Analysts' section
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
                    name = lines[0].strip()
                    # The firm might be on the next line
                    firm = lines[1].strip() if len(lines) > 1 else ''
                    # Clean and standardize the name
                    name = self.clean_and_standardize_name(name)
                    analysts.append({'name': name, 'firm': firm})
        else:
            print(f"Warning: 'Conference Call Participants' section not found in the transcript for analyst extraction.")
        return analysts

    def clean_and_standardize_name(self, name):
        """
        Clean and standardize the participant's name to 'F. Lastname' format.
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

        # Extract the first initial
        first_initial = first_word[0].upper() + '.'

        # Combine as 'F. Lastname' with Lastname capitalized
        standardized_name = f"{first_initial} {last_word.capitalize()}"

        return standardized_name.strip()

    def extract_and_split_section(self, company, call_id, company_info, date, text):
        """
        Extract and split both the 'Presentation' and 'Questions and Answers' sections from the text.
        """
        # Extract participants and analysts
        participants = self.extract_participants(text)
        analysts = self.extract_analysts(text)
        if not analysts:
            print(f"No analysts found in 'Conference Call Participants' for call ID: {call_id}")
        else:
            print(f"Analysts found for call ID {call_id}: {[analyst['name'] for analyst in analysts]}")

        # Create sets of names for matching
        analyst_names = set([analyst['name'] for analyst in analysts])
        manager_names = set([participant['name'] for participant in participants])

        # Determine if CEO participates
        ceo_participates = any(self.is_ceo(p['position']) for p in participants)

        # Store CEO and CFO names
        ceo_names = [p['name'] for p in participants if self.is_ceo(p['position'])]
        cfo_names = [p['name'] for p in participants if self.is_cfo(p['position'])]

        # Extract 'Presentation' section using original logic
        presentation_text = []
        presentation_section = self.extract_presentation_section_original(text, call_id)
        if presentation_section:
            presentation_section = self.clean_text_presentation(presentation_section)
            if self.method == 'sentences':
                presentation_text = self.split_text(presentation_section)
            else:
                presentation_text = self.split_text_by_visual_cues(presentation_section)
            # Final cleanup
            presentation_text = [self.remove_separator_line(para) for para in presentation_text]
            presentation_text = [para for para in presentation_text if para.strip()]
            presentation_text = self.remove_presentation_from_final_list(presentation_text)
            presentation_text = self.filter_short_elements(presentation_text)
        else:
            print(f"'Presentation' section not found for call ID: {call_id}")

        # Extract 'Questions and Answers' section
        participant_questions = []
        management_answers = []
        qa_section = self.extract_qa_section(text, call_id)
        if qa_section:
            participant_questions, management_answers = self.extract_questions_and_answers(
                qa_section, call_id, analyst_names, manager_names
            )
            # Remove the first and last elements assuming they are operator statements
            if len(participant_questions) >= 2:
                participant_questions = participant_questions[1:-1]
                print(f"Removed first and last operator statements from participant questions for call ID: {call_id}.")
            else:
                # If fewer than 2 questions, remove all to avoid including operator statements
                participant_questions = []
                print(f"Insufficient participant questions to remove operator statements for call ID: {call_id}.")

            # Final cleanup
            participant_questions = [self.remove_separator_line(para) for para in participant_questions]
            participant_questions = [para for para in participant_questions if para.strip()]
            participant_questions = self.filter_short_elements(participant_questions)

            management_answers = [self.remove_separator_line(para) for para in management_answers]
            management_answers = [para for para in management_answers if para.strip()]
            management_answers = self.filter_short_elements(management_answers)
        else:
            print(f"'Questions and Answers' section not found for call ID: {call_id}")

        if not presentation_text and not participant_questions and not management_answers:
            print(f"No relevant sections found for call ID: {call_id}")
            return None

        result = {
            'presentation_text': presentation_text,
            'participant_questions': participant_questions,
            'management_answers': management_answers,
            'participants': participants,
            'ceo_participates': int(ceo_participates),  # Ensure it's an integer (0 or 1)
            'ceo_names': ceo_names,
            'cfo_names': cfo_names
        }

        return result

    def extract_presentation_section_original(self, text, call_id):
        """
        Extract the 'Presentation' section from the transcript text using the original logic.
        """
        # Original logic as per the user's working script
        # Search for the beginning of the presentation
        match_presentation = re.search(r"(?<=\n)={1,}\nPresentation\n-{1,}(?=\n)", text)
        if match_presentation:
            start_index = match_presentation.end()
            # The presentation ends where the Q&A begins
            match_qanda = re.search(r"(?<=\n)={1,}\nQuestions and Answers\n-{1,}(?=\n)", text)
            if match_qanda:
                end_index = match_qanda.start()
                presentation_section = text[start_index:end_index]
            else:
                # If no Q&A, take till the end
                presentation_section = text[start_index:]
            return presentation_section.strip()
        else:
            print(f"'Presentation' section not found for call ID: {call_id}")
            return ''

    def clean_text_presentation(self, text):
        """
        Clean the 'Presentation' section using the original logic from the user's working script.
        """
        # Remove operator/editor statements
        # Pattern to find operator/editor statements
        pattern_operator = r'\n-{0,}-{0,}\n(?:Operator|Editor) {1,}\[[0-9]{1,3}\]\n-{0,}-{0,}'
        text = re.sub(pattern_operator, '\n', text, flags=re.IGNORECASE)

        # Remove speaker identification lines
        pattern_speaker = r'\n-{1,}\n[^\[\n]{1,} {1,}\[[0-9]{1,3}\]\n-{1,}\n'
        text = re.sub(pattern_speaker, '\n', text, flags=re.IGNORECASE)

        # Remove technical remarks
        pattern_technical = r'\([^\)]{1,}\)'
        text = re.sub(pattern_technical, '', text)

        # Remove multiple line breaks
        text = re.sub(r'\n{2,}', '\n', text)

        # Remove leading/trailing whitespaces
        text = text.strip()

        return text

    def extract_qa_section(self, text, call_id):
        """
        Extract the 'Questions and Answers' section from the transcript text.
        """
        # Adjusted pattern to match 'Questions and Answers' as a section heading with lines of '=' and '-'
        pattern = r'(?<=\n)={1,}\nQuestions and Answers\n-{1,}(?=\n)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            start_index = match.end()
            qa_section = text[start_index:]
        else:
            qa_section = ''
            print(f"'Questions and Answers' section not found for call ID: {call_id}")

        return qa_section.strip()

    def clean_text(self, text):
        """
        Clean the text by removing unwanted sections and patterns.
        """
        # For 'Presentation' section, use separate cleaning
        return text  # No additional cleaning here; handled in 'extract_and_split_section'

    def extract_questions_and_answers(self, qa_section, call_id, analyst_names, manager_names):
        """
        Extract questions from analysts and answers from managers in the 'Questions and Answers' section.
        Additionally, remove operator statements at the beginning and end of the Q&A section.
        """
        analyst_questions = []
        management_answers = []
        dropped_text = []

        # Remove technical remarks
        qa_section = re.sub(r'\((inaudible|technical difficulty|corrected by company after the call)\)', '', qa_section, flags=re.IGNORECASE)

        # Split the qa_section into a list of speaker segments
        qa_list = re.split(r'\n-{1,}\n(?=[^\[\n]+\[[0-9]+\]\n-{1,}\n)', qa_section)

        # Create variables to track whether the last segment was a question or an answer
        answer = 0
        question = 0
        operator = 0

        # Create variables to count the number of questions and answers
        answer_counter = 1
        question_counter = 1

        # Process each segment
        for k in range(len(qa_list)):
            # Split each segment into speaker_info and text
            speaker_text = re.split(r'(?<=\])\n-{1,}\n', qa_list[k])
            if len(speaker_text) < 2:
                continue  # Skip if format is not as expected
            speaker_info = speaker_text[0]
            speaker_name = speaker_info.split(", ")[0]
            speaker_name = speaker_name.replace("\n", "").strip()
            text = speaker_text[1]

            # Clean and standardize the speaker's name
            speaker_name = self.clean_and_standardize_name(speaker_name)

            # Check if the speaker is in manager_names or analyst_names
            if speaker_name in manager_names:
                # Process as management answer
                if k == 0:
                    # First element should be operator or analyst question
                    # Drop if not
                    dropped_text.append(f"{speaker_name.upper()}: {text}")
                elif answer == 0 and question_counter > answer_counter:
                    # Previous text was not an answer and we have more questions than answers
                    management_answers.append(f"ANSWER_{answer_counter}:\n{speaker_name.upper()}:\n{text}")
                    answer = 1
                    answer_counter += 1
                    question = 0
                    operator = 0
                else:
                    # Last segment was an answer
                    # Add to previous answer
                    if management_answers:
                        management_answers[-1] += f"\n{speaker_name.upper()}:\n{text}"
                    else:
                        # Edge case: answer without preceding question
                        management_answers.append(f"ANSWER_{answer_counter}:\n{speaker_name.upper()}:\n{text}")
                        answer_counter += 1
                    question = 0
                    operator = 0
            elif speaker_name.lower().startswith("operator") or speaker_name.lower().startswith("editor"):
                # It's an operator statement; drop it
                dropped_text.append(f"OPERATOR: {text}")
            else:
                # Speaker is an analyst
                # Check if the text is a 'thank you' or a question
                if (('thank' in text.lower() and len(text) < 100) or
                    (len(text) < 50 and '?' not in text) or
                    (k < len(qa_list) - 1 and
                     (qa_list[k + 1].lower().startswith("operator") or qa_list[k + 1].lower().startswith("editor")))):
                    # Likely a thank you or brief statement; drop it
                    dropped_text.append(f"{speaker_name.upper()}: {text}")
                else:
                    # It's a question
                    analyst_questions.append(f"QUESTION_{question_counter}:\n{speaker_name.upper()}:\n{text}")
                    question = 1
                    question_counter += 1
                    answer = 0
                    operator = 0

        # After processing, check whether number of questions and answers match
        if answer_counter != question_counter:
            print(f"The number of questions and answers does not match: {call_id}")

        # Debug: print collected questions and answers
        if analyst_questions:
            print(f"Collected {len(analyst_questions)} analyst questions for call ID {call_id}.")
        else:
            print(f"No analyst questions found for call ID {call_id}.")

        if management_answers:
            print(f"Collected {len(management_answers)} management answers for call ID {call_id}.")
        else:
            print(f"No management answers found for call ID {call_id}.")

        return analyst_questions, management_answers

    def remove_unwanted_sections(self, text):
        """
        Remove blocks like "TEXT version of Transcript", "Corporate Participants", "Conference Call Participants" from a transcript text.
        """
        pattern = r'(TEXT version of Transcript|Corporate Participants|Company Participants|Participants|Conference Call Participants|Analysts)\n=+\n(?:.*?\n)*?(?=\n=+\n|\Z)'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_text

    def remove_concluding_statements(self, text):
        """
        Remove concluding statements that might be present after the main content.
        """
        pattern = r'(Ladies and gentlemen, thank you for participating.*|This concludes today\'s program.*|You may all disconnect.*)'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def remove_pattern(self, text):
        """
        Remove patterns like "---- some text ----"
        """
        pattern = r'-{4,}.*?-{4,}'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text

    def remove_specific_string(self, text):
        """
        Forcefully remove "Presentation" from the text, regardless of where it appears.
        """
        # Remove "TEXT version of Transcript" case-insensitively
        text = re.sub(r"TEXT version of Transcript", '', text, flags=re.IGNORECASE)

        # Forcefully remove "Presentation" anywhere it appears at the start of a line, along with optional whitespace
        text = re.sub(r'(?i)^\s*Presentation\s*\n', '', text, flags=re.MULTILINE)

        return text.strip()

    def remove_separator_line(self, text):
        """
        Remove lines that are just a series of "=" or "-" characters
        """
        return re.sub(r'^\s*[-=]{3,}\s*$', '', text, flags=re.MULTILINE).strip()

    def split_text(self, text):
        """
        Split text into sentences or paragraphs, depending on the method specified.
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
        """
        return [element for element in text_list if element.strip().lower() != "presentation"]

    def filter_short_elements(self, text_list):
        """
        Remove elements that have fewer than 3 words from the list.
        """
        return [element for element in text_list if len(element.split()) >= 3]

    def is_ceo(self, position):
        """
        Determine if the position corresponds to a CEO.
        """
        for title_pattern in self.ceo_titles:
            if re.search(title_pattern, position, flags=re.IGNORECASE):
                return True
        return False

    def is_cfo(self, position):
        """
        Determine if the position corresponds to a CFO.
        """
        for title_pattern in self.cfo_titles:
            if re.search(title_pattern, position, flags=re.IGNORECASE):
                return True
        return False

    def extract_all_relevant_sections(self, ecc_sample, max_documents):
        """
        Extract all relevant sections from the ECC sample.
        Returns three lists: all_relevant_sections, all_relevant_questions, all_management_answers
        """
        all_relevant_sections = []
        all_relevant_questions = []
        all_management_answers = []
        document_count = 0

        for permco, calls in ecc_sample.items():
            for call_id, value in calls.items():
                if max_documents is not None and document_count >= max_documents:
                    break
                company_info = value.get('company_name', 'Unknown')
                date = value.get('date', 'Unknown')
                text = value.get('text_content', '')
                result = self.extract_and_split_section(permco, call_id, company_info, date, text)
                if result and (result.get('presentation_text') or result.get('participant_questions') or result.get('management_answers')):
                    # Collect presentation_text, participant_questions, and management_answers
                    if result.get('presentation_text'):
                        all_relevant_sections.extend(result['presentation_text'])
                    if result.get('participant_questions'):
                        all_relevant_questions.extend(result['participant_questions'])
                    if result.get('management_answers'):
                        all_management_answers.extend(result['management_answers'])

                    # Add the relevant data to 'value'
                    value['presentation_text'] = result['presentation_text'] if result.get('presentation_text') else []
                    value['participant_questions'] = result['participant_questions'] if result.get('participant_questions') else []
                    value['management_answers'] = result['management_answers'] if result.get('management_answers') else []
                    value['participants'] = result.get('participants', [])
                    value['ceo_participates'] = result.get('ceo_participates', False)
                    value['ceo_names'] = result.get('ceo_names', [])
                    value['cfo_names'] = result.get('cfo_names', [])

                    document_count += 1
                else:
                    print(f"No relevant sections found for call ID: {call_id}")

            if max_documents is not None and document_count >= max_documents:
                break

        return all_relevant_sections, all_relevant_questions, all_management_answers

    def split_text_by_visual_cues(self, text):
        """
        Split text into paragraphs by visual cues (e.g., blank lines, separators, etc.) and remove any unnecessary paragraphs.
        """
        pattern = r'\n\s*\n|\n[=]+\n|\n[-]+\n|\n\s{2,}\n|(?:^|\n)(?=[A-Z][a-z]+, [a-zA-Z\s]*[-]*[0-9]*)|\.\s*\n'
        paragraphs = re.split(pattern, text)
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        paragraphs = self.preprocess_paragraphs(paragraphs)
        return paragraphs

    def preprocess_paragraphs(self, paragraphs):
        """
        Remove unnecessary paragraphs from the list of paragraphs.
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
        """
        pattern = r"^[A-Z][a-zA-Z\. ]+, [A-Z][a-zA-Z\. ]+, [A-Za-z\.,&; ]+ - [A-Z]{2,}.*$"
        return re.match(pattern, paragraph.strip()) is not None

    def has_content_indicators(self, paragraph):
        """
        Check if a paragraph contains any of the following phrases which are indicators of a presentation section.
        """
        indicators = [
            "pleasure to", "turn the call over", "let me introduce", "i will now hand over",
            "welcome", "thank", "let's get started"
        ]
        return any(phrase in paragraph.lower() for phrase in indicators)

    def is_separator_line(self, paragraph):
        """
        Check if a paragraph is a separator line.
        """
        pattern = r"^[-=]{3,}$"
        return re.match(pattern, paragraph.strip()) is not None
