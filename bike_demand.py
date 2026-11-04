# Bike Sharing Demand Analysis
# Run: python bike_demand.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
hour = pd.read_csv("hour.csv")
day = pd.read_csv("day.csv")
hour["dteday"] = pd.to_datetime(hour["dteday"])
day["dteday"] = pd.to_datetime(day["dteday"])

os.makedirs("graphs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Check data quality
print("\nDATA CHECKS")
for name, df in [("Hourly", hour), ("Daily", day)]:
    print("\n", name, df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate IDs:", df["instant"].duplicated().sum())
    print("Incorrect rental totals:",
          (df["cnt"] != df["casual"] + df["registered"]).sum())

# Explore hourly demand
plt.figure(figsize=(9, 5))
hour.groupby("hr")["cnt"].mean().plot(marker="o")
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour")
plt.ylabel("Average rentals")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/hourly_demand.png")
plt.close()

# Compare working and non-working days
hour["day_type"] = np.where(hour["workingday"] == 1,
                            "Working day", "Non-working day")
plt.figure(figsize=(9, 5))
for name, group in hour.groupby("day_type"):
    group.groupby("hr")["cnt"].mean().plot(marker="o", label=name)
plt.title("Working Days vs Non-working Days")
plt.xlabel("Hour")
plt.ylabel("Average rentals")
plt.xticks(range(24))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/working_vs_nonworking.png")
plt.close()

# Compare seasons
season_avg = hour.groupby("season")["cnt"].mean()
plt.figure(figsize=(7, 5))
plt.bar(["Spring", "Summer", "Fall", "Winter"], season_avg.values)
plt.title("Average Rentals by Season")
plt.ylabel("Average rentals per hour")
plt.tight_layout()
plt.savefig("graphs/seasonal_demand.png")
plt.close()

# Compare weather conditions
weather_avg = hour.groupby("weathersit")["cnt"].mean()
plt.figure(figsize=(8, 5))
plt.bar(["Clear", "Mist/cloudy", "Light rain/snow", "Heavy rain/snow"],
        weather_avg.values)
plt.title("Average Rentals by Weather")
plt.ylabel("Average rentals per hour")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/weather_demand.png")
plt.close()

# Monthly trend using daily data
day["month"] = day["dteday"].dt.to_period("M").astype(str)
monthly_avg = day.groupby("month")["cnt"].mean()
plt.figure(figsize=(11, 5))
plt.plot(monthly_avg.index, monthly_avg.values, marker="o")
plt.title("Monthly Average Bike Rentals")
plt.xlabel("Month")
plt.ylabel("Average rentals per day")
plt.xticks(rotation=60)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/monthly_demand.png")
plt.close()

# Hypothesis test: rush-hour demand is higher on working days
rush_hours = [7, 8, 9, 16, 17, 18]
working = hour[hour["workingday"] == 1].copy()
working["rush"] = working["hr"].isin(rush_hours)

daily_avg = working.groupby("dteday").apply(
    lambda x: pd.Series({
        "rush": x.loc[x["rush"], "cnt"].mean(),
        "other": x.loc[~x["rush"], "cnt"].mean()
    }),
    include_groups=False
).dropna()

test = ttest_rel(daily_avg["rush"], daily_avg["other"])
rush_mean = daily_avg["rush"].mean()
other_mean = daily_avg["other"].mean()

print("\nHYPOTHESIS TEST")
print("Rush-hour average:", round(rush_mean, 2))
print("Other-hour average:", round(other_mean, 2))
print("P-value:", test.pvalue)

with open("outputs/hypothesis.txt", "w") as f:
    f.write(f"Rush-hour average: {rush_mean:.2f}\n")
    f.write(f"Other-hour average: {other_mean:.2f}\n")
    f.write(f"P-value: {test.pvalue:.6e}\n")

# Create model features
hour["hour_sin"] = np.sin(2 * np.pi * hour["hr"] / 24)
hour["hour_cos"] = np.cos(2 * np.pi * hour["hr"] / 24)
hour["month_sin"] = np.sin(2 * np.pi * hour["mnth"] / 12)
hour["month_cos"] = np.cos(2 * np.pi * hour["mnth"] / 12)
day_num = hour["dteday"].dt.day_of_year
hour["day_sin"] = np.sin(2 * np.pi * day_num / 365.25)
hour["day_cos"] = np.cos(2 * np.pi * day_num / 365.25)
hour["rush_hour"] = hour["hr"].isin(rush_hours).astype(int)

features = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum",
    "windspeed", "hour_sin", "hour_cos", "month_sin",
    "month_cos", "day_sin", "day_cos", "rush_hour"
]

# Use earlier dates for training and later dates for testing
dates = np.sort(hour["dteday"].unique())
split_date = dates[int(len(dates) * 0.80)]
train = hour[hour["dteday"] < split_date]
test_data = hour[hour["dteday"] >= split_date]

model = RandomForestRegressor(
    n_estimators=200, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model.fit(train[features], train["cnt"])
pred = model.predict(test_data[features])

mae = mean_absolute_error(test_data["cnt"], pred)
rmse = np.sqrt(mean_squared_error(test_data["cnt"], pred))
r2 = r2_score(test_data["cnt"], pred)
baseline = mean_absolute_error(
    test_data["cnt"], np.full(len(test_data), train["cnt"].mean())
)

print("\nMODEL RESULTS")
print("Training rows:", len(train))
print("Testing rows:", len(test_data))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))
print("Baseline MAE:", round(baseline, 2))

with open("outputs/model_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.4f}\nBaseline MAE: {baseline:.2f}\n")

# Save test predictions
results = test_data[["dteday", "hr", "cnt"]].copy()
results["predicted_cnt"] = pred
results.to_csv("outputs/test_predictions.csv", index=False)

# Actual vs predicted graph
sample = results.head(7 * 24)
plt.figure(figsize=(10, 5))
plt.plot(sample["cnt"].values, label="Actual")
plt.plot(sample["predicted_cnt"].values, label="Predicted")
plt.title("Actual vs Predicted Rentals (First Test Week)")
plt.xlabel("Hour")
plt.ylabel("Rentals")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png")
plt.close()

# Feature importance graph
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values().tail(10)
plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.title("Most Important Model Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("graphs/feature_importance.png")
plt.close()

# Save model for the Streamlit app
joblib.dump({"model": model, "features": features}, "outputs/model.joblib")

print("\nFinished. Check the graphs and outputs folders.")
print("Note: The model predicts hourly rentals, not station-level bike shortages.")
