# Convolutional Neural Network (CNN)
# app/models/cnn_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Flatten, Dense, UpSampling1D,Reshape,MaxPooling1D,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os



# create cnn model start
def generate_cnn_model(input_shape):
    model = Sequential()
    
    # Encoder
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    
    # Flatten and bottleneck layer
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))  # Bottleneck layer
    
    # Decoder
    model.add(Dense(8 * (input_shape[0] // 4), activation='relu'))
    model.add(Reshape((input_shape[0] // 4, 8)))
    model.add(UpSampling1D(2))
    model.add(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=1, kernel_size=2, activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='mse')
    return model
# create cnn model end


# train and save model start
def create_and_train_cnn_model(username,filepath):
    # Load the keystroke data
    data = pd.read_csv(filepath)
    key_out = data.drop('key',axis=1)
    key_out = key_out.drop('hold_time',axis=1)
    key_out = key_out.drop('flight_time',axis=1)
    # Use all data as features (no label column)
    X = key_out.values

    # Split data for testing (here using 80% training and 20% testing)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Train/test split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Reshape if necessary for the CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create the model
    model = generate_cnn_model(X_train.shape[1:])

    # Model summary
    # model.summary()
    
    # Train the model
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    model_dir = os.path.join('trained_models','model_files')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, f'{username}_keystroke_auth_model.keras')
    # Save the trained model
    model.save(model_filepath)
    
    return model,model_filepath
    
def cnn_authentication():
    pass

def cnn_registration(username,feauture_data):
    # Detect anomalies
    # Load the trained autoencoder model
    autoencoder = tf.keras.models.load_model(model_filepath)

    # Compute reconstruction error on test data
    reconstructions = autoencoder.predict(X_test)
    # reconstructions = autoencoder.predict(X)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    # mse = np.mean(np.power(X - reconstructions, 2), axis=1)

    # Define a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # For example, use the 95th percentile as the threshold

    # Predict anomalies
    anomalies = mse > threshold
    
    print("\n\n anomalies",anomalies)
    print("\n\n reconstructions:-",reconstructions)
    
    results = "fail"
    if len(anomalies)>0:
        if anomalies[0]:
            results = False
        else:
            results = True
    return results

# train and save model end