# Generative Adversarial Network (GAN)
# app/models/gan_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_generator_model(input_dim):
    # Initialize the generator model
    model = Sequential()
    
    # Add dense layers to the generator
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(input_dim, activation='tanh'))  # Output layer
    
    return model

def create_discriminator_model(input_dim):
    # Initialize the discriminator model
    model = Sequential()
    
    # Add dense layers to the discriminator
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    return model
