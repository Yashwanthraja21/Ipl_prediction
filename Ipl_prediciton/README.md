# IPL Match Winner Prediction System

A machine learning-based system that predicts the winner of IPL (Indian Premier League) cricket matches using historical match data and advanced analytics.

## Features

- Predicts match winners based on team composition and historical performance
- Interactive web interface built with Streamlit
- Uses historical IPL match data for training
- Implements machine learning algorithms for prediction

## Project Structure

```
.
├── data/
│   ├── matches.csv       # Historical match data
│   └── deliveries.csv    # Ball-by-ball details
├── src/
│   ├── app.py           # Streamlit web application
│   ├── data_preparation.py    # Data processing utilities
│   ├── ipl_predictor.py      # Core prediction logic
│   └── train_model.py        # Model training script
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
python -m streamlit run src/app.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- streamlit
- joblib
- requests
- beautifulsoup4

## License

This project is licensed under the MIT License - see the LICENSE file for details.