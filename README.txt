# Bike Sharing Demand Prediction

## 1. Project Overview

The objective of this project is to analyze bike-sharing data, understand rental patterns, predict hourly demand, and build a simple assistant for operations managers.

The main business question is whether the company should purchase more bikes or improve the distribution of its existing fleet.

**Dataset:**
- `hour.csv`: 17,379 hourly records.
- `day.csv`: 731 daily records.
- Data covers 2011–2012.

The target variable is `cnt`, which represents the total number of bike rentals per hour.

## 2. Data Analysis and Quality Checks

Both datasets were checked for:
- Missing values
- Duplicate records and IDs
- Negative rental counts
- Incorrect total rental counts

All these checks passed.

Some key observations:
- Average hourly rentals: 189.46
- Minimum hourly rentals: 1
- Maximum hourly rentals: 977
- Average rentals at 8 AM: approximately 359
- Average rentals at 5 PM: approximately 462

Exploratory graphs were generated to understand hourly, seasonal, monthly, and weather-related demand patterns.

## 3. Hypothesis Testing

**Hypothesis:** Bike rental demand during working-day rush hours is higher than during other hours.

- H0: Rush-hour demand is not higher than demand during other hours.
- H1: Rush-hour demand is higher.

Rush hours were defined as 7–9 AM and 4–6 PM.

A paired t-test was performed by comparing the average rush-hour and non-rush-hour demand for each working day.

**Results:**
- Rush-hour average: 385.77 rentals/hour
- Other-hour average: 127.40 rentals/hour
- P-value: 3.18 × 10⁻²¹⁰

The result provides strong statistical evidence that rush-hour demand is higher.

## 4. Demand Prediction Model

A **Random Forest Regressor** was used to predict hourly bike rentals.

**Features used:**
- Hour, month, season and year
- Weekday, holiday and working-day information
- Weather conditions
- Temperature, humidity and wind speed
- Cyclic time features
- Rush-hour indicator

The `casual` and `registered` columns were excluded because their sum is the target variable `cnt`. Including them would cause data leakage.

**Model training:**
- Training data: First 80% of dates
- Testing data: Final 20% of dates
- Algorithm: Random Forest Regressor
- Number of trees: 200

A chronological split was used to test the model on later dates rather than randomly mixing past and future observations.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | 55.59 |
| RMSE | 82.48 |
| R² Score | 0.8601 |
| Baseline MAE | 174.97 |

The model explains approximately 86% of the variation in test-set rental demand and performs substantially better than the mean-only baseline.

## 5. Natural Language Assistant

A Streamlit application was developed to demonstrate how an operations manager could ask questions about expected bike demand.

Example questions:
- How many bikes will we need at 8 AM on a working day in summer?
- Estimate demand at 8 AM on a rainy winter day.
- What is the demand at 6 PM on a hot weekend?

The application extracts information such as time, season and weather from the question and passes the resulting inputs to the trained model.

This is a lightweight, rule-based prototype, not an LLM.

## 6. Assumptions and Limitations

- Historical rental patterns are useful for estimating future demand.
- The `cnt` column represents hourly rental demand.
- Weather values are normalized according to the dataset documentation.
- When weather information is missing from a question, the application uses default values.
- The model predicts hourly rental volume, not the number of bikes required at a specific station.

**Important business limitation:**

The dataset does not contain station-level inventory, failed rental requests, bike availability, or rebalancing information.

Therefore, the model cannot confirm whether customers were unable to rent bikes because of shortages.

### Business Recommendation

The company should not purchase additional bikes based only on this model.

It should first use station-level demand forecasting and improve bike redistribution. Additional bikes should be considered if shortages continue even after improving redistribution.

## 7. Project Structure

```text
Bike-Sharing-Demand-Prediction/
│
├── bike_demand.py
├── app.py
├── hour.csv
├── day.csv
├── requirements.txt
│
├── graphs/
├── outputs/
│   ├── model.joblib
│   ├── model_metrics.txt
│   ├── hypothesis.txt
│   └── test_predictions.csv
│
└── screenshots/
```

## 8. How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the analysis and train the model:

```bash
python bike_demand.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

The training script generates graphs, evaluation results, test predictions, and the trained model.

## 9. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Streamlit
- Joblib