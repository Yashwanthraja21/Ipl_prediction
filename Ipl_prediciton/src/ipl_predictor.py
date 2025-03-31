import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X, y):
    """Train the IPL match prediction model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def predict_winner(model, match_data):
    """Predict the winner of a match using the trained model."""
    try:
        # Make prediction
        prediction = model.predict(match_data)
        probability = model.predict_proba(match_data)
        
        # Get the predicted winner and its probability
        winner = prediction[0]
        win_prob = np.max(probability[0])
        
        return winner, win_prob
        
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def save_model(model, filename='ipl_model.joblib'):
    """Save the trained model to a file."""
    try:
        joblib.dump(model, filename)
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")

def load_model(filename='ipl_model.joblib'):
    """Load a trained model from a file."""
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")