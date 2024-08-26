import os
import json
import time
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from file_handling import FileHandler  # Import the FileHandler class
from text_processing import TextProcessor  # Import the TextProcessor class
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT  # Import KeyBERT
from utils import print_configuration


class BertopicModel:
    def __init__(self, config):
        self.config = config
        self.model_save_path = config["model_save_path"]
        self.modeling_type = config.get("modeling_type", "regular") #default
        self.doc_chunk_size = config.get("doc_chunk_size", 5000) #default
        self.topic_model = None
        self.model = self._load_sentence_transformer(config["finbert_model_path"])

    def _load_sentence_transformer(self, model_path):
        print("Loading SentenceTransformer model...")
        if not os.path.exists(model_path):
            raise ValueError(f"The specified model path does not exist: {model_path}")
        return SentenceTransformer(model_path)

    def _initialize_bertopic_model(self):
        
        if self.modeling_type in ["zeroshot", "iterative_zeroshot"]:
            print("Initializing zeroshot BERTopic model...")
            return BERTopic(
                embedding_model=self.config["finbert_model_path"],
                min_topic_size=self.config["min_topic_size"],
                zeroshot_topic_list=self.config["zeroshot_topic_list"],
                zeroshot_min_similarity=self.config["zeroshot_min_similarity"],
                representation_model=KeyBERTInspired()
            )
        else:
            return BERTopic(
                embedding_model=self.model,
                min_topic_size=self.config["min_topic_size"],
                representation_model=KeyBERTInspired()
            )

    def train(self, docs):
        if self.modeling_type in ["iterative", "iterative_zeroshot"]:
            self._train_iterative(docs)
        else:
            self._train_regular(docs)

    def _train_regular(self, docs):
        self.topic_model = self._initialize_bertopic_model()
        print("Training BERTopic model...")
        start_time = time.time()

        # Fit the BERTopic model
        topics, probs = self.topic_model.fit_transform(docs)

        end_time = time.time()

        # Print information about the training process
        print(f"BERTopic model trained on {len(docs)} sections.")
        print(f"Number of topics generated: {len(set(topics))}")
        print(f"Training time: {end_time - start_time:.2f} seconds.")

        # Save the BERTopic model to the specified path
        print("Saving BERTopic model...")
        self.topic_model.save(self.model_save_path)
        print(f"BERTopic model saved to {self.model_save_path}.")

    def _train_iterative(self, docs):
        print("Initializing iterative BERTopic model...")
        doc_chunks = [docs[i:i+self.doc_chunk_size] for i in range(0, len(docs), self.doc_chunk_size)]

        base_model = self._initialize_bertopic_model().fit(doc_chunks[0])

        for chunk in doc_chunks[1:]:
            print("Merging new documents into the base model...")
            new_model = self._initialize_bertopic_model().fit(chunk)
            updated_model = BERTopic.merge_models([base_model, new_model])

            # Print the newly discovered topics
            nr_new_topics = len(set(updated_model.topics_)) - len(set(base_model.topics_))
            new_topics = list(updated_model.topic_labels_.values())[-nr_new_topics:]
            print("The following topics are newly found:")
            print(f"{new_topics}\n")

            # Update the base model
            base_model = updated_model

        self.topic_model = base_model

        print("Saving the final merged BERTopic model...")
        self.topic_model.save(self.model_save_path)
        print(f"Final BERTopic model saved to {self.model_save_path}.")

    def extract_keywords(self):
        print("Initializing KeyBERT for keyword extraction...")
        kw_model = KeyBERT(model=self.model)

        print("Extracting keywords for each topic...")
        for topic in range(len(self.topic_model.get_topics())):
            topic_words = self.topic_model.get_topic(topic)
            if topic_words:
                topic_keywords = kw_model.extract_keywords(
                    " ".join([word[0] for word in topic_words]), 
                    keyphrase_ngram_range=tuple(self.config["keybert_params"]["keyphrase_ngram_range"]),
                    stop_words=self.config["keybert_params"]["stop_words"],
                    use_maxsum=self.config["keybert_params"]["use_maxsum"],
                    use_mmr=self.config["keybert_params"]["use_mmr"],
                    diversity=self.config["keybert_params"]["diversity"],
                    top_n=self.config["keybert_params"]["top_n"]
                )
                print(f"Topic {topic}: {topic_keywords}")
            else:
                print(f"Topic {topic} has no significant words.")


def main():
    # Load configuration from config.json  
    print("Loading configuration...")
    with open('C:/Users/nikla/OneDrive/Dokumente/winfoMaster/Masterarbeit/bertopic_ecc/config.json', 'r') as config_file:
        config = json.load(config_file)
    print_configuration(config)

    # Extract variables from the config
    index_file_ecc_folder = config["index_file_ecc_folder"]
    folderpath_ecc = config["folderpath_ecc"]
    sample_size = config["sample_size"]
    document_split = config["document_split"]
    section_to_analyze = config["section_to_analyze"]
    max_documents = config["max_documents"]

    # Initialize FileHandler and TextProcessor with the imported configuration
    print("Initializing file handler and text processor...")
    file_handler = FileHandler(index_file_path=config["index_file_path"], folderpath_ecc=folderpath_ecc)
    text_processor = TextProcessor(method=document_split, section_to_analyze=section_to_analyze)

    # Create the sample and extract relevant sections
    print("Reading index file and creating ECC sample...")
    index_file = file_handler.read_index_file()
    ecc_sample = file_handler.create_ecc_sample(sample_size)
    all_relevant_sections = text_processor.extract_all_relevant_sections(ecc_sample, max_documents)

    if not all_relevant_sections:
        print("No relevant sections found to fit BERTopic.")
        return

    # Instantiate and train the BERTopic model
    bertopic_model = BertopicModel(config)
    bertopic_model.train(all_relevant_sections)

    # Extract and print keywords
    bertopic_model.extract_keywords()


if __name__ == "__main__":
    main()
