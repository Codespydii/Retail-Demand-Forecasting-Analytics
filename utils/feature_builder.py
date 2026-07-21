import pandas as pd
import streamlit as st


@st.cache_data
def load_featured_data():
    """
    Load the engineered dataset only once.
    """
    df = pd.read_csv("data/processed/featured_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_features(
    store,
    prediction_date,
    promo,
    school_holiday,
    state_holiday,
):
    """
    Build one feature row for prediction.
    """

    df = load_featured_data()

    # Get history for selected store
    store_df = df[df["Store"] == store].copy()

    if store_df.empty:
        raise ValueError(f"No data found for Store {store}")

    # Latest available record
    latest = store_df.sort_values("Date").iloc[-1].copy()

    prediction_date = pd.to_datetime(prediction_date)

    # --------------------------
    # Update date-based features
    # --------------------------

    latest["Year"] = prediction_date.year
    latest["Month"] = prediction_date.month
    latest["Quarter"] = prediction_date.quarter
    latest["WeekOfYear"] = prediction_date.isocalendar().week
    latest["Day"] = prediction_date.day
    latest["DayOfYear"] = prediction_date.dayofyear
    latest["DayOfWeek"] = prediction_date.dayofweek + 1

    latest["IsWeekend"] = int(prediction_date.dayofweek >= 5)

    latest["MonthStart"] = int(prediction_date.is_month_start)
    latest["MonthEnd"] = int(prediction_date.is_month_end)

    latest["DateOrdinal"] = prediction_date.toordinal()

    # Cyclical Encoding
    import numpy as np

    latest["MonthSin"] = np.sin(
        2 * np.pi * latest["Month"] / 12
    )

    latest["MonthCos"] = np.cos(
        2 * np.pi * latest["Month"] / 12
    )

    latest["WeekdaySin"] = np.sin(
        2 * np.pi * latest["DayOfWeek"] / 7
    )

    latest["WeekdayCos"] = np.cos(
        2 * np.pi * latest["DayOfWeek"] / 7
    )

    # --------------------------
    # User Inputs
    # --------------------------

    latest["Promo"] = promo
    latest["SchoolHoliday"] = school_holiday
    latest["StateHoliday"] = state_holiday

    # Remove target
    latest = latest.drop(labels=["Sales", "Date"])

    return pd.DataFrame([latest])