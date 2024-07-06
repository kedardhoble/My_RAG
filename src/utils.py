# Utility functions can be defined here
import re

def clean_text(text):
    # Example: Remove special characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def tokenize(text):
    return text.split()
