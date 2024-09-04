import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
def create_train_random_forest_model(username,filepath):
    # Save the models
    model_dir = os.path.join("trained_models","random_forest_model_files")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    rf_model_filepath = os.path.join(model_dir,f'{username}_keystroke_random_forest_model.pkl')
    rf_scaler_filepath = os.path.join(model_dir,f'{username}_scaler_random_forest_model.pkl')
    data = pd.read_csv(filepath)
    
    features = np.array(data['flight_time'],data['delay_time'])
    features = features.reshape(1,-1)
    print("Feature:\n",features,"\n\n","Feature shape:\n",features.shape,"\n\n")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    rf_model = RandomForestClassifier()
    rf_model.fit(features,np.array([1]))
    
    joblib.dump(rf_model, rf_model_filepath)
    joblib.dump(scaler, rf_scaler_filepath)
    
    
    
    # # auth
    # features_auth = np.array(data['flight_time'],data['delay_time'])
    # features_auth = features.reshape(1,-1)
    
    # rfml_load_model = joblib.load(rf_model_filepath)
    # rfml_load_scaler = joblib.load(rf_scaler_filepath)
    
    # feature_values = rfml_load_scaler.transform(features_auth)
    
    # predictions = rfml_load_model.predict(feature_values)
    # authenticate = np.mean(predictions)
    # print("predictions", predictions,"\n\n","auth",bool(authenticate))
    
    return rf_model_filepath,rf_scaler_filepath,scaler,rf_model