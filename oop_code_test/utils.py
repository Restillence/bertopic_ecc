# utils.py
import json

def print_configuration(config):
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
