"""
Run after training:
    python bike_demand.py
    streamlit run app.py

This is a lightweight, rule-based natural-language demo (not an LLM).
It uses the model saved by bike_demand.py at outputs/model.joblib.
"""

import re
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Bike Demand Assistant", page_icon="🚲")
st.title("🚲 Bike Demand Assistant")
st.caption("Ask an operations-style question about expected hourly rental demand.")


@st.cache_resource
def load_bundle():
    return joblib.load("outputs/model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        "Trained model not found. Keep this app.py in the project folder, "
        "then run `python bike_demand.py` first."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


def parse_question(question):
    text = question.lower()

    # Date: accepts YYYY-MM-DD. Defaults to a date in the dataset period.
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if date_match:
        try:
            selected_date = pd.to_datetime(
                date_match.group(1), errors="raise"
            ).date()
        except ValueError:
            raise ValueError("Please enter a valid date in YYYY-MM-DD format.")
    else:
        selected_date = date(2012, 6, 15)

    # Time: supports 8 AM, 5 pm, 'at 17', or 'hour 17'.
    time_match = re.search(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", text)
    if time_match:
        hour = int(time_match.group(1)) % 12
        if time_match.group(2) == "pm":
            hour += 12
    else:
        time_24_match = re.search(r"\b(?:at|hour)\s*(\d{1,2})\b", text)
        hour = int(time_24_match.group(1)) if time_24_match else 8

    if not 0 <= hour <= 23:
        raise ValueError("Please specify an hour from 0 to 23, or use AM/PM.")

    # Bike-sharing dataset season codes:
    # 1 = spring, 2 = summer, 3 = fall, 4 = winter.
    month_to_season = {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4,
    }
    if "summer" in text:
        season = 2
    elif "fall" in text or "autumn" in text:
        season = 3
    elif "winter" in text:
        season = 4
    elif "spring" in text:
        season = 1
    else:
        season = month_to_season[selected_date.month]

    # Weather codes: 1 = clear, 2 = mist/cloudy, 3 = light rain/snow.
    if any(word in text for word in ["rain", "snow", "storm"]):
        weather = 3
    elif any(word in text for word in ["mist", "cloud", "fog"]):
        weather = 2
    else:
        weather = 1

    is_weekend = any(
        word in text for word in ["weekend", "saturday", "sunday"]
    )
    holiday = int("holiday" in text)
    workingday = 0 if is_weekend or holiday else 1

    # Approximate normalized weather values if not explicitly specified.
    temp, atemp, humidity, windspeed = 0.50, 0.48, 0.60, 0.20
    if "cold" in text:
        temp, atemp = 0.25, 0.24
    elif "hot" in text:
        temp, atemp = 0.80, 0.78
    if "humid" in text:
        humidity = 0.85
    if "windy" in text:
        windspeed = 0.40

    return (
        selected_date, hour, season, weather, workingday, holiday,
        temp, atemp, humidity, windspeed
    )


def make_model_row(values):
    (
        selected_date, hour, season, weather, workingday, holiday,
        temp, atemp, humidity, windspeed
    ) = values

    timestamp = pd.Timestamp(selected_date)
    month = selected_date.month
    weekday = timestamp.weekday()
    year_code = 1 if selected_date.year >= 2012 else 0
    day_of_year = timestamp.dayofyear

    row = {
        "season": season,
        "yr": year_code,
        "mnth": month,
        "hr": hour,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temp,
        "atemp": atemp,
        "hum": humidity,
        "windspeed": windspeed,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        # Feature names used by the current training script.
        "day_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "rush_hour": int(hour in [7, 8, 9, 16, 17, 18]),

        # Keep aliases too, so the app remains compatible with earlier
        # versions of the training script.
        "dayofyear_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "dayofyear_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18]),
    }

    # Preserve the exact feature names and order used during model training.
    missing_features = [name for name in bundle["features"] if name not in row]
    if missing_features:
        raise ValueError(
            "The app is missing model features: " + ", ".join(missing_features)
            + ". Please check that app.py and bike_demand.py are from the same version."
        )
    return pd.DataFrame([row])[bundle["features"]]


question = st.text_input(
    "Ask a question",
    "How many bikes will we need at 8 AM on a working day in summer?"
)

if st.button("Estimate demand", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            parsed_values = parse_question(question)
            model_input = make_model_row(parsed_values)
            prediction = float(bundle["model"].predict(model_input)[0])
            prediction = max(0, prediction)

            st.metric("Estimated hourly rentals", f"{prediction:,.0f}")
            st.write(
                f"**Parsed scenario:** {parsed_values[0]} at "
                f"{parsed_values[1]:02d}:00"
            )
            st.write(
                f"**Season code:** {parsed_values[2]}  |  "
                f"**Weather code:** {parsed_values[3]}"
            )
            st.info(
                "This predicts hourly rental volume. It does not estimate "
                "how many bikes must be physically available at one specific dock."
            )
            st.caption(
                "If weather details are omitted, the prototype uses default "
                "weather values. Treat the result as a planning estimate, "
                "not a live availability guarantee."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")


st.divider()
st.subheader("Try asking")
st.write("- How many bikes will we need at 5 PM on a working day?")
st.write("- Estimate demand at 8 AM on a rainy winter day.")
st.write("- What is the demand at 6 PM on a hot weekend?")
st.write("- Estimate demand on 2012-08-15 at 17.")

st.caption(
    "Prototype limitation: this demo recognizes a small set of natural-language "
    "patterns. A production version should use validated weather forecasts and "
    "station-level inventory data. This is a rule-based demo, not an LLM."
)
