import os
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "input.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_dataset():
    """Downloads the Tiny Shakespeare dataset if it doesn't already exist locally."""
    if not os.path.exists(FILE_PATH):
        print(f"Downloading Tiny Shakespeare dataset from {URL}...")
        response = requests.get(URL)
        response.raise_for_status()
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Successfully saved dataset to {FILE_PATH} ({len(response.text):,} characters).")
    else:
        print(f"Dataset already exists at {FILE_PATH}.")
    
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset summary: {len(text):,} characters, {len(set(text))} unique characters.")

if __name__ == "__main__":
    download_dataset()
