import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('chatbot_model.h5')

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Example: Predicting the intent for a new sentence
input_text = ["Hello, how are you?"]

# Step 1: Tokenize and pad the input text
input_sequence = tokenizer.texts_to_sequences(input_text)
input_padded = pad_sequences(input_sequence, padding='post', maxlen=20)  # Ensure maxlen matches training

# Step 2: Make prediction using the trained model
prediction = model.predict(input_padded)

# Step 3: Convert the model's output into the corresponding intent label
predicted_class = label_encoder.inverse_transform([prediction.argmax()])

# Output the predicted intent
print(f"Predicted Intent: {predicted_class[0]}")
