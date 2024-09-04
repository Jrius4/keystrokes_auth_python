# app/routes/auth_routes.py
import os
from flask import Blueprint, request, jsonify
from app.utils.data_processing import preprocess_keystroke_data, save_keystroke_data, generate_features, get_processed_data,save_keystroke_featured_data
from app.utils.model_training import train_and_save_model
from app.paradigms_v1.cnn_model import create_cnn_model
from app.paradigms_v1.lstm_model import create_lstm_model
from app.paradigms_v1.randomForest_model import train_random_forest_model
from app.models.models import UserKeystrokes, TrainedModels
from app.extensions import db
import numpy as np
import joblib
from sqlalchemy.exc import IntegrityError


from app.paradigms_v2.cnn_model import create_and_train_cnn_model
from app.paradigms_v2.randomForest_model import create_train_random_forest_model

# Define a blueprint for the authentication routes
auth_ep = Blueprint('auth_ep', __name__)
# 

# authenticate
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
    
    try:
        # Store paths in the database
        user_keystroke = UserKeystrokes(username=username, keystroke_path=keystroke_csv_path, features_path=features_csv_path)
        db.session.add(user_keystroke)
        db.session.commit()
        
        trained_model = TrainedModels(username=username, model_type='CNN', model_path=model_path)
        db.session.add(trained_model)
        db.session.commit()
    except db.IntegrityError:
        return "Username already exists!", 400
    finally:
        db.session.commit()
    
    # Make predictions using the model
    prediction = model.predict(processed_data)
    
    # Check the prediction result and respond accordingly
    authenticated = bool(prediction > 0.5)
    
    return jsonify({'authenticated': authenticated})
# register
@auth_ep.route('/register', methods=['POST'])
def register():
    pass

# cnn
# register
@auth_ep.route('/register-cnn', methods=['POST'])
def registerCnn():
    username = request.json['username']
    keystrokes = request.json['keystrokes']
    
    
    # save raw keydata
    raw_keystroke_csv_path = save_keystroke_data(username,keystrokes)
    featured_path = generate_features(username,raw_keystroke_csv_path)
    model,model_path = create_and_train_cnn_model(username,featured_path)
    try:
        # Store paths in the database
        user_keystroke = UserKeystrokes(username=username, keystroke_path=raw_keystroke_csv_path, features_path=featured_path)
        db.session.add(user_keystroke)
        db.session.commit()
        
        trained_model = TrainedModels(username=username, model_type='CNNV2', model_path=model_path)
        db.session.add(trained_model)
        db.session.commit()
    except db.IntegrityError:
        return "Username already exists!", 400
    finally:
        db.close()
    data = get_processed_data(featured_path)
    features = np.array(data['delay_time'])
    # prediction = model.predict(features)
    # print("prediction: ",prediction)
    # Check the prediction result and respond accordingly
    # authenticated = bool(prediction > 0.5)
    # Compute reconstruction error on test data
    reconstructions = model.predict(features)
    # reconstructions = autoencoder.predict(X)
    mse = np.mean(np.power(features - reconstructions, 2), axis=1)
    # mse = np.mean(np.power(X - reconstructions, 2), axis=1)

    # Define a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # For example, use the 95th percentile as the threshold

    # Predict anomalies
    anomalies = mse > threshold
    
    print("\n\n anomalies",anomalies)
    print("\n\n reconstructions:-",reconstructions)
    
    authenticated = "fail"
    if len(anomalies)>0:
        if anomalies[0]:
            authenticated = False
        else:
            authenticated = True
    print("\n\n is user authenicate: ", authenticated)
    
    return jsonify({'authenticated': authenticated})
    
# authenticate
@auth_ep.route('/authenticate-cnn', methods=['POST'])
def authenticateCnn():
    pass
# mlp
# register
@auth_ep.route('/authenticate-mlp', methods=['POST'])
def registerMlp():
    username = request.json['username']
    keystrokes = request.json['keystrokes']
    
    
    # save raw keydata
    raw_keystroke_csv_path = save_keystroke_data(username,keystrokes)
    featured_path = generate_features(username,raw_keystroke_csv_path)
    model,model_path = create_and_train_cnn_model(username,featured_path)
    try:
        # Store paths in the database
        user_keystroke = UserKeystrokes(username=username, keystroke_path=raw_keystroke_csv_path, features_path=featured_path)
        db.session.add(user_keystroke)
        db.session.commit()
        
        trained_model = TrainedModels(username=username, model_type='CNNV2', model_path=model_path)
        db.session.add(trained_model)
        db.session.commit()
    except db.IntegrityError:
        return "Username already exists!", 400
    finally:
        db.close()
    data = get_processed_data(featured_path)
    features = np.array(data['delay_time'])
    # prediction = model.predict(features)
    # print("prediction: ",prediction)
    # Check the prediction result and respond accordingly
    # authenticated = bool(prediction > 0.5)
    # Compute reconstruction error on test data
    reconstructions = model.predict(features)
    # reconstructions = autoencoder.predict(X)
    mse = np.mean(np.power(features - reconstructions, 2), axis=1)
    # mse = np.mean(np.power(X - reconstructions, 2), axis=1)

    # Define a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # For example, use the 95th percentile as the threshold

    # Predict anomalies
    anomalies = mse > threshold
    
    print("\n\n anomalies",anomalies)
    print("\n\n reconstructions:-",reconstructions)
    
    authenticated = "fail"
    if len(anomalies)>0:
        if anomalies[0]:
            authenticated = False
        else:
            authenticated = True
    print("\n\n is user authenicate: ", authenticated)
    
    return jsonify({'authenticated': authenticated})

# authenticate
@auth_ep.route('/register-mlp', methods=['POST'])
def authenticateMlp():
    pass
# lstm
# register
@auth_ep.route('/authenticate-lstm', methods=['POST'])
def registerLstm():
    pass
# authenticate
@auth_ep.route('/register-lstm', methods=['POST'])
def authenticateLstm():
    pass
# gan
# register
@auth_ep.route('/authenticate-gan', methods=['POST'])
def registerGan():
    pass
# authenticate
@auth_ep.route('/register-gan', methods=['POST'])
def authenticateGan():
    pass
# rf
# register
@auth_ep.route('/register-rf', methods=['POST'])
def registerRf():
    username = request.json['username']
    keystrokes = request.json['keystrokes']
    
    
    # save raw keydata
    raw_keystroke_csv_path = save_keystroke_data(username,keystrokes)
    featured_path = generate_features(username,raw_keystroke_csv_path)
    rf_model_filepath,rf_scaler_filepath,scaler,rf_model = create_train_random_forest_model(username,featured_path)
    data = get_processed_data(featured_path)
     # auth
    features = np.array(data['flight_time'],data['delay_time'])
    features_auth = np.array(data['flight_time'],data['delay_time'])
    features_auth = features.reshape(1,-1)
    
    try:
        # Store paths in the database
        user_keystroke = UserKeystrokes(username=username, keystroke_path=raw_keystroke_csv_path, features_path=featured_path)
        db.session.add(user_keystroke)
        db.session.commit()
        
        trained_model = TrainedModels(username=username, model_type='randomforest', model_path=rf_model_filepath)
        db.session.add(trained_model)
        trained_model = TrainedModels(username=username, model_type='randomforest_scaler', model_path=rf_scaler_filepath)
        db.session.add(trained_model)
        db.session.commit()
    except IntegrityError:
        return "Username already exists!", 409
    finally:
        db.session.rollback()
    
    rfml_load_model = joblib.load(rf_model_filepath)
    rfml_load_scaler = joblib.load(rf_scaler_filepath)
    
    feature_values = rfml_load_scaler.transform(features_auth)
    
    predictions = rfml_load_model.predict(feature_values)
    authenticate = np.mean(predictions)
    if bool(authenticate):
        return jsonify({'authenticated': bool(authenticate)}) 
# authenticate
@auth_ep.route('/authenticate-rf', methods=['POST'])
def authenticateRf():
    username = request.json['username']
    keystrokes = request.json['keystrokes']
    
    
    # save raw keydata
    raw_keystroke_csv_path = save_keystroke_data(username,keystrokes)
    featured_path = generate_features(username,raw_keystroke_csv_path)
    # rf_model_filepath,rf_scaler_filepath,scaler,rf_model = create_train_random_forest_model(username,featured_path)
    data = get_processed_data(featured_path)
     # auth
    features = np.array(data['flight_time'],data['delay_time'])
    features_auth = np.array(data['flight_time'],data['delay_time'])
    features_auth = features.reshape(1,-1)
    
    try:
        trainedModel = TrainedModels.query.filter(TrainedModels.username==username,TrainedModels.model_type=='randomforest').first()
        trainedModelScaler = TrainedModels.query.filter(TrainedModels.username== username,TrainedModels.model_type=='randomforest_scaler').first()
        if trainedModel:
            rf_model_filepath = trainedModel.model_path
            rf_scaler_filepath = trainedModelScaler.model_path
            rfml_load_model = joblib.load(rf_model_filepath)
            rfml_load_scaler = joblib.load(rf_scaler_filepath)
            
            feature_values = rfml_load_scaler.transform(features_auth)
            
            predictions = rfml_load_model.predict(feature_values)
            authenticate = np.mean(predictions)
            if bool(authenticate):
                return jsonify({'authenticated': bool(authenticate)}),200 
            
        else:
            return jsonify({"error": "User not found"}), 404
    except IntegrityError:
        return "Username NOT found!", 400
    
    finally:
        db.session.rollback()
    
    rfml_load_model = joblib.load(rf_model_filepath)
    rfml_load_scaler = joblib.load(rf_scaler_filepath)
    
    feature_values = rfml_load_scaler.transform(features_auth)
    
    predictions = rfml_load_model.predict(feature_values)
    authenticate = np.mean(predictions)
    if bool(authenticate):
        return jsonify({'authenticated': bool(authenticate)}) 