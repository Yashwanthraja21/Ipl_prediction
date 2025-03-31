import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_match_data(matches_file='data/matches.csv'):
    """Load and preprocess the matches dataset."""
    try:
        matches_df = pd.read_csv(matches_file)
        return matches_df
    except Exception as e:
        raise Exception(f"Error loading match data: {str(e)}")

def prepare_match_data(team1, team2):
    """Prepare feature vector for prediction based on two teams."""
    # Create basic features
    match_features = {
        'team1': team1,
        'team2': team2,
        'toss_winner': team1,  # Default assumption
        'toss_decision': 'bat',  # Default assumption
        'venue': 'neutral',  # Default venue
    }
    
    # Convert to DataFrame
    match_df = pd.DataFrame([match_features])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        match_df[column] = label_encoders[column].fit_transform(match_df[column])
    
    return match_df

def calculate_team_stats(matches_df, team):
    """Calculate historical statistics for a given team."""
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
    
    total_matches = len(team_matches)
    wins = len(team_matches[team_matches['winner'] == team])
    win_rate = wins / total_matches if total_matches > 0 else 0
    
    return {
        'total_matches': total_matches,
        'wins': wins,
        'win_rate': win_rate
    }

def get_head_to_head_stats(matches_df, team1, team2):
    """Calculate head-to-head statistics between two teams."""
    h2h_matches = matches_df[
        ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
        ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
    ]
    
    team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
    team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
    
    return {
        'total_matches': len(h2h_matches),
        f'{team1}_wins': team1_wins,
        f'{team2}_wins': team2_wins
    }