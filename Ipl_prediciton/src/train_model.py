import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_preparation import load_match_data
from ipl_predictor import train_model, save_model

def prepare_training_data(matches_df):
    """Prepare the training dataset from matches data."""
    # Select relevant features
    features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    target = 'winner'
    
    # Create feature matrix
    X = matches_df[features].copy()
    y = matches_df[target].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for column in features:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Encode target variable
    y = LabelEncoder().fit_transform(y)
    
    return X, y

def main():
    try:
        # Load match data
        print("Loading match data...")
        matches_df = load_match_data()
        
        # Prepare training data
        print("Preparing training data...")
        X, y = prepare_training_data(matches_df)
        
        # Train model
        print("Training model...")
        model, accuracy = train_model(X, y)
        print(f"Model training completed. Accuracy: {accuracy:.2%}")
        
        # Save model
        print("Saving model...")
        save_model(model)
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")

if __name__ == '__main__':
    main()