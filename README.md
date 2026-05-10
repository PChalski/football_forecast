# Football Forecast

Football probabilities prediction for 1X2 outcomes using Logisistic Regression and Poisson distribution based model.
Betting simulation, checking whether model is able to beat bookmaker, predicting future matches, identifying possible value bets.

Data collected from https://www.football-data.co.uk/

## Models

- **LR** - Logistic Regression trained on rolling features (form, goal difference, home advantage)
- **Poisson** - Poisson distribution based probability model using historical team scoring data to calculate Poisson distribution expected value parameter
- **Random** - Random betting

## Features

- Form difference (5, 10, 15 matches)
- Goal difference (5, 10, 15 matches)
- Home advantage
- League position difference (used mainly to subsitute form/goal diff features at the start of the season where data is incomplete)

## Setup

```bash
pip install -r requirements.txt
python create_db.py
streamlit run app.py
```

## Usage

1. **Training** - Train LR and Poisson models on selected league/seasons.
2. **Insights** - View model performance metrics such as logloss, brier score, ROC AUC.
3. **Betting Simulation** - Test model's peformance, check if model is able to win against odds provider.
4. **Predict** - Make predictions for future matches, identify possible value bets.

## Results (Betting Simulation)

Logistic Regression model performs better than random and Poisson models, but slightly worse than bookmaker's predictions.
Simple machine learning model using only few features is unable to beat bookmaker's odds in a long run, although it can be used as a simple benchmark 
to predict future odds.