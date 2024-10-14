# utils.py
import json
from bertopic import BERTopic

def print_configuration(config):
    """
    Print the configuration dictionary.

    Parameters:
    - config (dict): The configuration dictionary to be printed.
    """
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def load_bertopic_model(model_path):
    """
    Load the trained BERTopic model from a file.

    Parameters:
    - model_path (str): The path to the saved BERTopic model file.

    Returns:
    - BERTopic: The loaded BERTopic model.
    """
    topic_model = BERTopic.load(model_path)
    print(f"BERTopic model loaded from {model_path}")
    return topic_model

def heartbeat():
    """
    Prints a heartbeat message to the console every 5 minutes.
    Runs indefinitely until the main program exits.
    """
    while True:
        time.sleep(300)  # 300 seconds = 5 minutes
        print("[Heartbeat] The script is still running...")