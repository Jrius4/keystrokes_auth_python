import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os



# create Multi-Layer Perceptron (MLP) model start
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# create Multi-Layer Perceptron (MLP) model end


# train and save mlp model start
def preprocess_mlp_data(data):
    X = data.drop('key', axis=1).values  # Features
    y = data.drop('key',axis=1)['delay_time'].values  # Labels
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
def create_and_train_mlp_model(username,filepath):
    # Load the keystroke data
    data = pd.read_csv(filepath)
    X_train, X_test, y_train, y_test = preprocess_mlp_data(data)
    
    print("X_train.shape: ",X_train.shape,"X_test.shape: ",X_test.shape)
    
    model = create_mlp_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the trained model
    model_dir = os.path.join('trained_models','model_files')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, f'{username}_keystroke_auth_mlp_model.keras')
    # Save the trained model
    model.save(model_filepath)
    model.summary
    
    return model, model_filepath
    
    # # Detect anomalies
    # # Load the trained autoencoder model
    # autoencoder = tf.keras.models.load_model(model_filepath)

    # # Compute reconstruction error on test data
    # reconstructions = autoencoder.predict(X_test)
    # # reconstructions = autoencoder.predict(X)
    # mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    # # mse = np.mean(np.power(X - reconstructions, 2), axis=1)

    # # Define a threshold for anomaly detection
    # threshold = np.percentile(mse, 95)  # For example, use the 95th percentile as the threshold

    # # Predict anomalies
    # anomalies = mse > threshold
    
    # # Output results
    # print("\n\nreconstructions: ",reconstructions)
    
    
    # # if mse > threshold:
    # #     print(f"Anomaly detection threshold: {threshold} (imposter)")
    # #     print("\n")
    # # else:
    # #     print(f"Anomaly detection threshold: {threshold} (genuine user)")
    # #     print("\n")

    # # Output results
    # # for i, is_anomaly in enumerate(anomalies):
    # #     if is_anomaly:
    # #         print(f"Test sample {i} is an anomaly (likely an impostor).")
    # #         print("\n")
    # #     else:
    # #         print(f"Test sample {i} is normal (likely a genuine user).")
    # #         print("\n")
            
    # print("\n\nanomalies: ", anomalies)
    # results = "fail"
    # if len(anomalies)>0:
    #     if anomalies[0]:
    #         results = False
    #     else:
    #         results = True
    # print("\n\n is user authenicate: ", results)
    
    
#train model mlp end