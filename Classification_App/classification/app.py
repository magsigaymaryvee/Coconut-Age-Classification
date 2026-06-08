# app.py
import streamlit as st
import os
import joblib
from screens import ModelSelector, HomeInput, OutputPage
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

st.set_page_config(page_title="🌴 Coconut Age Classifier", page_icon="🥥", layout="centered")

# ---------------------------
# Load Models (FIXED PATH)
# ---------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    svm_normal_path = os.path.join(model_dir, "coconut_age_classifier.pkl")
    svm_smote_path = os.path.join(model_dir, "coconut_age_classifier_smote.pkl")

    svm_normal = joblib.load(svm_normal_path)
    svm_smote = joblib.load(svm_smote_path)

    cnn_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    return svm_normal, svm_smote, cnn_model

svm_normal, svm_smote, cnn_model = load_models()

# ---------------------------
# App Navigation
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "model_selector"

if st.session_state.page == "model_selector":
    ModelSelector.show_page(svm_normal, svm_smote, cnn_model)

elif st.session_state.page == "home_input":
    HomeInput.show_page()

elif st.session_state.page == "output_page":
    OutputPage.show_page()