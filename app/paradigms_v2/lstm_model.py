import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
# train and create Long Short-Term Memory (LSTM) Network model start
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def process_lstm_data(data):
    X = data.drop('key', axis=1).values  # Features
    y = data.drop('key',axis=1)['delay_time'].values  # Labels
    
    # standarize feautures
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape input data for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    return X_train, X_test, y_train, y_test,X_scaled

def lstm_prediction_users_auth(username,filepath):
    # Load the saved model
    data = pd.read_csv(filepath)
    key_features =  data.drop('key',axis=1).values  # Labels
    model_dir = os.path.join("trained_models","lstm_model_files")
    lstm_model_filepath = os.path.join(model_dir,f'{username}_keystroke_lstm_model.keras')
    lstm_model = tf.keras.models.load_model(lstm_model_filepath)
    
    # Load the scaler
    lstm_scaler_path = os.path.join(model_dir,f'{username}_scaler_numpy.npy')
    lstm_scaler_mean = np.load(lstm_scaler_path, allow_pickle=True)
    lstm_scaler_var = np.load(lstm_scaler_path.replace(".npy","_var.npy"), allow_pickle=True)
    if not isinstance(lstm_scaler_mean, np.ndarray) or not isinstance(lstm_scaler_var, np.ndarray):
        print("Failed to load scalar parameters")
        return
    print("lstm_scaler_mean:",lstm_scaler_mean,"lstm_scaler_var:",lstm_scaler_var)
    scaler = StandardScaler()
    scaler.mean_ = lstm_scaler_mean
    scaler.var_ = lstm_scaler_var
    scaler.scale_ = np.sqrt(lstm_scaler_var)
    
    scaler.n_samples_seen_ = 0
    key_features = scaler.fit_transform(key_features)
    
    key_features = np.reshape(key_features, (key_features.shape[0], 1, key_features.shape[1]))  # Reshape for LSTM
    
    predictions = lstm_model.predict(key_features)
    authenticated = (predictions[0][0] > 0.5)
    
    # Process new data
    #...
    
    # Apply scaler
    #...
    
    # Predict the delay time
    #...
    
    # Return the prediction
    
    print("lstm prediction: ",np.mean(predictions),"authenticated",authenticated)
    
    return authenticated,predictions
    
    
def create_and_train_lstm_model(username,filepath):
    data = pd.read_csv(filepath)
    
    X_train, X_test, y_train,y_test,X_scaled = process_lstm_data(data)
    lstm_model = build_lstm_model((1, X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    # Save the models
    model_dir = os.path.join("trained_models","lstm_model_files")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lstm_model_filepath = os.path.join(model_dir,f'{username}_keystroke_lstm_model.keras')
    lstm_model.save(lstm_model_filepath)
    
    # Save the model and scaler for later use
    print("Scaled Mean",np.mean(X_scaled),"X_scaled Var",np.var(X_scaled))
    lstm_scaler_path = os.path.join(model_dir,f'{username}_scaler_numpy.npy')
    np.save(lstm_scaler_path,np.mean(X_scaled))
    np.save(lstm_scaler_path.replace(".npy","_var.npy"),np.var(X_scaled))
    
    lstm_model.summary
    return lstm_model,lstm_model_filepath
    # lstm_prediction_users_auth(username,filepath)
# train and create Long Short-Term Memory (LSTM) Network model end