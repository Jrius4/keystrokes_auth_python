# app/utils/model_training.py
from tensorflow.keras.optimizers import Adam
import os

def train_and_save_model(model, data, username, epochs=10, batch_size=32):
    # Compile the model with a binary cross-entropy loss function and accuracy metric
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model on the keystroke data
    model.fit(data, data, epochs=epochs, batch_size=batch_size)
    
    # Save the trained model to a file and return the file path
    model_path = f'models/{username}_model.h5'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    return model_path
