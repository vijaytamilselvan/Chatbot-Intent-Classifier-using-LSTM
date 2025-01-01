import pickle
import json
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('Chatbot-Intent-Classifier-using-LSTM/data/chatbot_model.h5')

# Load the tokenizer
with open('Chatbot-Intent-Classifier-using-LSTM/data/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the label encoder
with open('Chatbot-Intent-Classifier-using-LSTM/data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('Chatbot-Intent-Classifier-using-LSTM/data/intents.json', 'r') as file:
    intents = json.load(file)

# Function to predict intent from input text
def predict_intent(input_text):
    # Step 1: Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, padding='post', maxlen=20)  # Ensure maxlen matches training

    # Step 2: Make prediction using the trained model
    prediction = model.predict(input_padded)

    # Step 3: Convert the model's output into the corresponding intent label
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])

    return predicted_class[0]

def get_response(predicted_intent):
    # Loop through the intents to find the corresponding response
    for intent in intents:
        if intent['tag'] == predicted_intent:
            return random.choice(intent['responses'])  # Return a random response from the list

    return "Sorry, I didn't understand that."

# Loop to continuously take input and predict intent
while True:
    # Accept input from the user
    input_text = input("Enter your message (or type 'exit' to quit): ")

    # Exit condition
    if input_text.lower() == 'exit':
        print("Exiting the chatbot.")
        break
    
    # Get prediction for the input text
    predicted_intent = predict_intent(input_text)

    response = get_response(predicted_intent)
    # Output the predicted intent
    print(f"Predicted Intent: {predicted_intent}")
    print(f"Response: {response}")
