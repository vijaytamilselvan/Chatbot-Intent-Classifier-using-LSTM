import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(json_file_path):
    """
    Loads the intents JSON file and preprocesses it into a pandas DataFrame.

    Args:
        json_file_path (str): Path to the intents.json file.
    
    Returns:
        df (DataFrame): A pandas DataFrame with columns 'tag', 'patterns', and 'responses'.
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Initialize a dictionary to store flattened data
    dic = {"tag": [], "patterns": [], "responses": []}

    # Loop through each intent in the data
    for intent in data["intents"]:
        tag = intent["tag"]
        patterns = intent["patterns"]
        responses = intent["responses"]

        # For each pattern, store the tag and associated responses as a list
        for pattern in patterns:
            dic['tag'].append(tag)
            dic['patterns'].append(pattern)
            dic['responses'].append(responses)  # Store the entire list of responses

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(dic)
    return df


def preprocess_data(df):
    """
    Preprocesses the text data by tokenizing, padding, and encoding labels.

    Args:
        df (DataFrame): A pandas DataFrame containing 'patterns' and 'tag' columns.
    
    Returns:
        X: Padded input sequences.
        y: Encoded output labels.
        tokenizer: The fitted tokenizer object.
        label_encoder: The fitted label encoder object.
    """
    # Tokenize the text data
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(df['patterns'])
    sequences = tokenizer.texts_to_sequences(df['patterns'])
    
    # Pad sequences
    X = pad_sequences(sequences, padding='post')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['tag'])
    
    return X, y, tokenizer, label_encoder
