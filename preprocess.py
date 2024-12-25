import re
from InpRespPairs import input_output_pairs

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to the dataset
preprocessed_pairs = [(preprocess_text(inp), preprocess_text(resp)) for inp, resp in input_output_pairs]

# print(f"Preprocessed Sample Pair: {preprocessed_pairs[0]}")


#! WHEN YOU FEED YOUR MODEL WITH OTHER TYPES OF DATA FOR SCALING, USE BELOW APPROACH;

# import pickle

# with open("input_output_pairs.pkl", "rb") as f:
#     input_output_pairs = pickle.load(f)
