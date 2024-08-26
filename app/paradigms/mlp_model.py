# Multi-Layer Perceptron (MLP)
# app/models/mlp_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp_model(input_dim):
    # Initialize the MLP model
    model = Sequential()
    
    # Add dense layers to the MLP
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    return model
