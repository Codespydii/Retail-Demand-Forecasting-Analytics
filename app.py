import streamlit as st
from utils.predictor import predict
from utils.feature_builder import build_features

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🏪",
    layout="wide"
)

# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.title("🏪 Retail Demand Forecasting Dashboard")
st.markdown(
    "Predict daily retail sales using an optimized XGBoost Machine Learning model."
)

st.divider()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Prediction Settings")

store = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=1115,
    value=1
)

prediction_date = st.sidebar.date_input(
    "Prediction Date"
)

promo = st.sidebar.selectbox(
    "Promotion",
    [0, 1]
)

school_holiday = st.sidebar.selectbox(
    "School Holiday",
    [0, 1]
)

state_holiday = st.sidebar.selectbox(
    "State Holiday",
    ["0", "a", "b", "c"]
)

predict_button = st.sidebar.button("🚀 Predict Sales")

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.info("""
**Algorithm:** XGBoost

**MAE:** 269.64

**RMSE:** 418.95

**R² Score:** 0.9882
""")

st.divider()

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if predict_button:

    try:

        # ------------------------------
        # Validation
        # ------------------------------

        if store < 1 or store > 1115:
            st.error("Please enter a valid Store ID.")
            st.stop()

        st.info(f"""
### Current Prediction

- **Store:** {store}
- **Date:** {prediction_date}
- **Promotion:** {"Yes" if promo else "No"}
- **School Holiday:** {"Yes" if school_holiday else "No"}
- **State Holiday:** {state_holiday}
""")

        # ------------------------------
        # Build Features
        # ------------------------------

        features = build_features(
            store=store,
            prediction_date=prediction_date,
            promo=promo,
            school_holiday=school_holiday,
            state_holiday=state_holiday
        )

        if "Open" in features.columns:
              if features.iloc[0]["Open"] == 0:
                   st.warning("⚠️ Store appears to be closed for the selected date."
                              )
        # ------------------------------
        # Prediction
        # ------------------------------

        prediction = predict(features)[0]

        st.session_state.prediction_history.append({
            "Store": store,
            "Prediction Date": prediction_date,
            "Promotion": "Yes" if promo else "No",
            "School Holiday": "Yes" if school_holiday else "No",
            "State Holiday": state_holiday,
            "Predicted Sales": round(float(prediction), 2)
        })

        st.success("Prediction Generated Successfully!")

        # ----------------------------------------------------
        # KPI Cards
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "💰 Predicted Sales",
                f"₹ {prediction:,.0f}"
            )

        with col2:

            st.metric(
                "🤖 Model",
                "XGBoost"
            )

        with col3:

            st.metric(
                "📊 R² Score",
                "0.9882"
            )

        st.divider()

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Prediction Details",
            "⚙️ Generated Features",
            "📜 Prediction History",
            "ℹ️ Model Information"
        ])
        # ----------------------------------------------------
        # Tab 1 - Prediction Details
        # ----------------------------------------------------

        with tab1:

            st.subheader("Prediction Summary")

            summary = {
                "Field": [
                    "Store ID",
                    "Prediction Date",
                    "Promotion",
                    "School Holiday",
                    "State Holiday"
                ],
                "Value": [
                    store,
                    prediction_date,
                    "Yes" if promo else "No",
                    "Yes" if school_holiday else "No",
                    state_holiday
                ]
            }

            st.table(summary)

            st.metric(
                label="Estimated Sales",
                value=f"₹ {prediction:,.2f}"
            )

        # ----------------------------------------------------
        # Tab 2 - Generated Features
        # ----------------------------------------------------

        with tab2:

            st.subheader("Generated Feature Vector")

            st.write(
                "The following engineered features were generated and supplied to the trained XGBoost model."
            )

            st.dataframe(
                features,
                use_container_width=True
            )

            csv = features.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Feature Vector (CSV)",
                data=csv,
                file_name="generated_features.csv",
                mime="text/csv"
            )

            st.divider()

            st.subheader("Feature Statistics")

            st.dataframe(
                features.describe().T,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Tab 3 - Prediction History
        # ----------------------------------------------------

        with tab3:

            st.subheader("Prediction History")

            if st.session_state.prediction_history:

                st.dataframe(
                    st.session_state.prediction_history,
                    use_container_width=True
                )

            else:

                st.info("No predictions have been made yet.")

        # ----------------------------------------------------
        # Tab 4 - Model Information
        # ----------------------------------------------------

        with tab4:

            st.subheader("About the Model")

            st.markdown("""
### 📌 Project Overview

This application predicts daily retail sales for Rossmann stores using an optimized Machine Learning pipeline.

The prediction workflow automatically generates engineered features before passing them to the trained XGBoost model.

---

### 🤖 Model Used

- XGBoost Regressor
- RandomizedSearchCV for Hyperparameter Tuning
- Scikit-learn Pipeline

---

### 📊 Model Performance

| Metric | Score |
|--------|-------|
| MAE | **269.64** |
| RMSE | **418.95** |
| R² Score | **0.9882** |

---

### ⚙️ Engineered Features

The model uses:

- Calendar Features
- Lag Features
- Rolling Mean Features
- Rolling Standard Deviation
- Customer Features
- Competition Features
- Promotion Features
- Store Metadata
- Cyclic Date Encoding

---

### 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit

---

### 📁 Project Modules

- Feature Builder
- Prediction Engine
- Streamlit Dashboard
- Session History
- Feature Export
- Input Validation
""")
            
    # ----------------------------------------------------
    # Error Handling
    # ----------------------------------------------------

    except Exception as e:

        st.error("❌ Prediction Failed")

        with st.expander("View Error Details"):
            st.exception(e)

# ----------------------------------------------------
# Landing Page
# ----------------------------------------------------

else:

    st.info(
        "👈 Configure the prediction settings from the sidebar and click **Predict Sales** to generate a forecast."
    )

    st.subheader("📌 About This Project")

    st.markdown("""
This application predicts **daily retail sales** using an optimized **XGBoost Regression Model** trained on the Rossmann Store Sales dataset.

The dashboard demonstrates an end-to-end Machine Learning workflow, including feature engineering, model inference, and interactive visualization.

---

### 🚀 Key Features

- Store-specific sales prediction
- Automatic feature engineering
- Calendar-based features
- Lag & Rolling statistics
- Competition & Promotion features
- Interactive Streamlit dashboard
- Download generated feature vector
- Prediction history tracking
- Input validation
- Professional dashboard interface

---

### 📊 Model Performance

| Metric | Value |
|--------|-------:|
| MAE | **269.64** |
| RMSE | **418.95** |
| R² Score | **0.9882** |

---

### 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit

---

### 📂 Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Hyperparameter Tuning
6. Model Evaluation
7. Streamlit Deployment
""")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.divider()

st.caption(
    "🏪 Retail Demand Forecasting Dashboard | Built with Streamlit, XGBoost & Scikit-learn"
)