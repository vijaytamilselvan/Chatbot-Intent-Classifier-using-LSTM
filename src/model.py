# Add LSTM model definition here

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization

def create_model(vocab_size, output_dim, input_length, num_classes):
    """
    Creates and compiles the LSTM model.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        output_dim (int): Dimension of the embedding layer output.
        input_length (int): Length of the input sequences.
        num_classes (int): Number of output classes (tags).
    
    Returns:
        model: A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim, input_length=input_length, mask_zero=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(32))
    model.add(LayerNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
