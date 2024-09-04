# app/utils/data_processing.py
import numpy as np
import pandas as pd
import os
import csv

import pandas as pd


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

# get processed date
def get_processed_data(filepath):
    return pd.read_csv(filepath)


# generate features start
def generate_features(username,filepath):
    # Generate features here

    process_dir = os.path.join('processed_data','processed_features')
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
    processed_filepath = os.path.join(process_dir, f'{username}_processed_featured_data.csv')
    # Load the raw keystroke data
    raw_data = pd.read_csv(filepath)

    # Sort data by timestamp to ensure correct order
    raw_data = raw_data.sort_values(by='time')

    # Initialize lists to store computed features
    hold_times = []
    flight_times = []
    delay_times = []

    # Initialize variables to keep track of previous key events
    prev_keyup_time = None
    prev_keydown_time = None
    prev_key = None

    # Process the raw keystroke data
    for index, row in raw_data.iterrows():
        key = row['key']
        event = row['event']
        timestamp = row['time']
        
        if event == 'keydown':
            # Calculate delay time (time between the previous keydown and current keydown)
            if prev_keydown_time is not None:
                delay_time = timestamp - prev_keydown_time
                delay_times.append({'key': key, 'delay_time': delay_time})
            
            # Update the previous keydown timestamp
            prev_keydown_time = timestamp
        
        elif event == 'keyup':
            # Calculate hold time (time between keydown and keyup of the same key)
            hold_time = timestamp - prev_keydown_time
            hold_times.append({'key': key, 'hold_time': hold_time})
            
            # Calculate flight time (time between the previous keyup and the current keydown)
            if prev_keyup_time is not None:
                flight_time = timestamp - prev_keyup_time
                flight_times.append({'key': key, 'flight_time': flight_time})
            
            # Update the previous keyup timestamp
            prev_keyup_time = timestamp

    # Convert the lists to DataFrames
    hold_times_df = pd.DataFrame(hold_times)
    flight_times_df = pd.DataFrame(flight_times)
    delay_times_df = pd.DataFrame(delay_times)
    # print("hold_times: ",hold_times)
    # print("\n")
    # print("flight_times: ", flight_times)
    # print("\n")

    # print("delay_times: ",delay_times)
    # print("\n")

    # Merge all the DataFrames on the 'key' column
    features_df = pd.merge(hold_times_df, flight_times_df, on='key', how='outer')
    features_df = pd.merge(features_df, delay_times_df, on='key', how='outer')
    
    print("\n")
    print("features_df: \n",features_df)

    # Save the features to a CSV file
    features_df.to_csv(processed_filepath, index=False)

    print(f"Keystroke features saved to '{processed_filepath}'")
    """
    Explanation of the Script

    Loading and Sorting:
        The script starts by loading the raw keystroke data from a CSV file and sorting it by timestamp to ensure the events are processed in chronological order.

    Feature Computation:
        Hold Time: The script calculates the time a key is held down by finding the difference between the keydown and keyup events for the same key.
        Flight Time: The time between releasing one key and pressing the next key is computed.
        Delay Time: The time between pressing two consecutive keys is calculated.

    Data Merging:
        The computed features are stored in separate lists, which are then converted to pandas DataFrames.
        These DataFrames are merged on the key column to form a single DataFrame containing all the features.
    """
    return processed_filepath



def preprocess_keystroke_data(data):
    print('Data: Shape\n')
    print(np.array(data).shape)
    # Convert the raw data into a numpy array and reshape it for model input
    # return np.array(data).reshape(-1, 28, 28, 1)
    press_times = []
    release_times = []
    
    for k in data:
        if k['event'] == 'keydown':
            press_times.append(k['time'])
        elif k['event'] == 'keyup':
            release_times.append(k['time'])
    
    # Flight time (time between press and release of the same key)
    flight_times = np.array(release_times) - np.array(press_times)
    
    # Delay time (time between release of one key and press of the next key)
    delay_times = np.array(press_times[1:]) - np.array(release_times[:-1])
    
    return np.concatenate([flight_times, delay_times])
    
    # return np.array(data).reshape(-1, 1)

# def save_keystroke_data(username, keystrokes):
#     # Save keystrokes to a CSV file and return the file path
#     # path = f'data/keystrokes/{username}_keystrokes.csv'
#     # os.makedirs(os.path.dirname(path), exist_ok=True)
#     # with open(path, 'w', newline='') as file:
#     #     writer = csv.writer(file)
#     #     writer.writerows(keystrokes)
#     # return path
#     df = pd.DataFrame(keystrokes)
#     raw_dir = os.path.join('data', 'raw')
#     if not os.path.exists(raw_dir):
#         os.makedirs(raw_dir)
#     filepath = os.path.join(raw_dir,f'{username}.csv')
#     df.to_csv(filepath,index=False)
#     return filepath

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
