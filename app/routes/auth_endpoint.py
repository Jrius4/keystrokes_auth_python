# app/routes/auth_routes.py
import os
from flask import Blueprint, request, jsonify
from app.utils.data_processing import preprocess_keystroke_data, save_keystroke_data, save_features, save_keystroke_featured_data
from app.utils.model_training import train_and_save_model
from app.paradigms.cnn_model import create_cnn_model
from app.paradigms.lstm_model import create_lstm_model
from app.paradigms.randomForest_model import train_random_forest_model
from app.models.models import UserKeystrokes, TrainedModels
from app.extensions import db

# Define a blueprint for the authentication routes
auth_ep = Blueprint('auth_ep', __name__)

@auth_ep.route('/authenticate', methods=['POST'])
def authenticate():
    # Get username and keystroke data from the request
    username = request.json['username']
    keystrokes = request.json['keystrokes']
    
    # Save the raw keystroke data as a CSV file and get the file path
    keystroke_csv_path = save_keystroke_data(username, keystrokes)
    
    # Extract features from the keystrokes and save them as a CSV file
    features_csv_path = save_keystroke_featured_data(username,keystroke_csv_path)
    
    # Train a Random Forest model using the keystroke data
    rf_model, rf_accuracy= train_random_forest_model(features_csv_path) # this passes
    print(f"Training Random Forest: '{rf_accuracy:.4f}'")
    
    # Preprocess the keystroke data
    processed_data = preprocess_keystroke_data(keystrokes)
    
    # kip rich
    print(f'processed_data shape: {processed_data.shape}',processed_data)
    kp_processed = processed_data.reshape(-1,1)
    
    print(f'processed_data shape: {kp_processed.shape}',kp_processed)
    
    # Extract features from the keystrokes and save them as a CSV file
    # features_csv_path = save_features(username, processed_data)
    
    
    
    # Choose a model based on your requirement (e.g., CNN)
    model = create_cnn_model(input_shape=(28, 28, 1))
    model_path = train_and_save_model(model, kp_processed, username)
    
    # Store paths in the database
    user_keystroke = UserKeystrokes(username=username, keystroke_path=keystroke_csv_path, features_path=features_csv_path)
    db.session.add(user_keystroke)
    db.session.commit()
    
    trained_model = TrainedModels(username=username, model_type='CNN', model_path=model_path)
    db.session.add(trained_model)
    db.session.commit()
    
    # Make predictions using the model
    prediction = model.predict(processed_data)
    
    # Check the prediction result and respond accordingly
    authenticated = bool(prediction > 0.5)
    
    return jsonify({'authenticated': authenticated})
