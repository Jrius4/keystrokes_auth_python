# app/utils/data_processing.py
import numpy as np
import pandas as pd
import os
import csv

def preprocess_keystroke_data(data):
    print('Data: Shape\n')
    print(np.array(data).shape)
    # Convert the raw data into a numpy array and reshape it for model input
    # return np.array(data).reshape(-1, 28, 28, 1)
    press_times = []
    release_times = []
    
    for k in data:
        if k['action'] == 'keydown':
            press_times.append(k['timestamp'])
        elif k['action'] == 'keyup':
            release_times.append(k['timestamp'])
    
    # Flight time (time between press and release of the same key)
    flight_times = np.array(release_times) - np.array(press_times)
    
    # Delay time (time between release of one key and press of the next key)
    delay_times = np.array(press_times[1:]) - np.array(release_times[:-1])
    
    return np.concatenate([flight_times, delay_times])
    
    return np.array(data).reshape(-1, 1)

def save_keystroke_data(username, keystrokes):
    # Save keystrokes to a CSV file and return the file path
    # path = f'data/keystrokes/{username}_keystrokes.csv'
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    # with open(path, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(keystrokes)
    # return path
    df = pd.DataFrame(keystrokes)
    raw_dir = os.path.join('data', 'raw')
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    filepath = os.path.join(raw_dir,f'{username}.csv')
    df.to_csv(filepath,index=False)
    return filepath

def save_keystroke_featured_data(username, filepath):
    feature_dir = os.path.join('data','featured')
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    featured_filepath = os.path.join(feature_dir, f'{username}.csv')
    df = pd.read_csv(filepath)
    # Extract features (e.g., flight time, delay time)
    hold_times = []
    flight_times = []
    last_release_time = None

    for i in range(len(df)):
        if df.iloc[i]['event'] == 'keydown':
            release_idx = df[(df['key'] == df.iloc[i]['key']) & (df['event'] == 'keyup')].index
            if not release_idx.empty:
                hold_time = df.loc[release_idx[0], 'time'] - df.iloc[i]['time']
                hold_times.append(hold_time)
            else:
                hold_times.append(None)
            if last_release_time is not None:
                flight_time = df.iloc[i]['time'] - last_release_time
                flight_times.append(flight_time)
            else:
                flight_times.append(None)
        
        if df.iloc[i]['event'] == 'keyup':
            last_release_time = df.iloc[i]['time']

    df['hold_time'] = pd.Series(hold_times)
    df['flight_time'] = pd.Series(flight_times)
    df_filtered = df[df['event'] == 'keydown'].reset_index(drop=True)
    df_filtered.to_csv(featured_filepath, index=False)
    return featured_filepath

def save_features(username, features):
    # Save features to a CSV file and return the file path
    path = f'data/features/{username}_features.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(features)
    return path
