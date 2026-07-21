try:
    import joblib
except ImportError:
    import pickle as joblib
import streamlit as st


@st.cache_resource
def load_model():
    """
    Load the trained model only once.
    Streamlit caches it, so it isn't reloaded on every interaction.
    """
    model = joblib.load("models/tuned_model.pkl")
    return model


def predict(input_df):
    """
    Make predictions using the trained pipeline.
    """
    model = load_model()
    prediction = model.predict(input_df)
    return prediction