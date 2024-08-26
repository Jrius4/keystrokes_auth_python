# Long Short-Term Memory (LSTM) Network
# app/models/lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    # Initialize the LSTM model
    model = Sequential()
    
    # Add LSTM layers
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    
    # Add the output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    return model
