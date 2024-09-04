from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def train_random_forest_model(filepath):
    # Load the keystroke data from the CSV file
    df = pd.read_csv(filepath)
    
    # Separate the features (flight time, delay time) and the target (keystroke type)
    X = df[['hold_time', 'flight_time']].fillna(0)
    y = df['key']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model on the testing set
    y_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    
    return rf_model, rf_accuracy