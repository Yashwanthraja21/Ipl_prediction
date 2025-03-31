import streamlit as st
import pandas as pd
import joblib
from data_preparation import prepare_match_data
from ipl_predictor import predict_winner

# Set page config
st.set_page_config(
    page_title="IPL Match Winner Predictor",
    page_icon="🏏",
    layout="wide"
)

# Title and description
st.title("IPL Match Winner Prediction System 🏏")
st.markdown("""
Predict the winner of an IPL match based on team composition and historical performance.
This model uses machine learning to analyze past match data and make predictions.
""")

# Load the pre-trained model
try:
    model = joblib.load('ipl_model.joblib')
except:
    st.error("Error: Model file not found. Please ensure the model is trained and saved.")
    st.stop()

# Create two columns for team selection
col1, col2 = st.columns(2)

# Team selection dropdowns
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
    'Rajasthan Royals', 'Sunrisers Hyderabad'
]

with col1:
    team1 = st.selectbox('Select Team 1', teams, index=0)

with col2:
    # Filter out team1 from options for team2
    team2_options = [team for team in teams if team != team1]
    team2 = st.selectbox('Select Team 2', team2_options, index=0)

# Add a predict button
if st.button('Predict Winner'):
    try:
        # Prepare match data
        match_data = prepare_match_data(team1, team2)
        
        # Make prediction
        winner, probability = predict_winner(model, match_data)
        
        # Display results
        st.success(f"Predicted Winner: {winner}")
        st.info(f"Win Probability: {probability:.2%}")
        
        # Display additional insights
        st.markdown("### Match Insights")
        st.write(f"• {winner} has a {probability:.2%} chance of winning this match")
        st.write(f"• Based on historical head-to-head performance and recent form")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and Machine Learning</p>
</div>
""", unsafe_allow_html=True)