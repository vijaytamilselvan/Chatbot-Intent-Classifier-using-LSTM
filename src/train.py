# Add training logic here

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the data
with open('chatbot-intent-classifier/data/intents.json', 'r') as f:
    data = json.load(f)

dic = {"tag": [], "patterns": [], "responses": []}
for intent in data["intents"]:
    tag = intent["tag"]
    patterns = intent["patterns"]
    responses = intent["responses"]
    
    for pattern in patterns:
        dic['tag'].append(tag)
        dic['patterns'].append(pattern)
        dic['responses'].append(responses)

df = pd.DataFrame.from_dict(dic)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['patterns'])
ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

# Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['tag'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, mask_zero=True))
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save('chatbot_model.h5', save_format='keras')

# Save the tokenizer
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model, Tokenizer, and Label Encoder saved successfully.")
