import streamlit as st
import torch
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier import load_model, predict, CLASS_NAMES
from rag import build_vectorstore, get_answer

st.set_page_config(
    page_title="Galaxy Classifier",
    page_icon="🌌",
    layout="centered"
)

st.title("🌌 Galaxy Morphology Classifier")
st.markdown("Upload a galaxy image and get a classification with a detailed explanation powered by RAG.")

@st.cache_resource
def load_resources():
    model = load_model("models/galaxy_classifier.pth")
    vectorstore = build_vectorstore()
    return model, vectorstore

with st.spinner("Loading model and knowledge base..."):
    model, vectorstore = load_resources()

st.success("Ready! Upload a galaxy image below.")

uploaded_file = st.file_uploader("Choose a galaxy image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying galaxy..."):
        galaxy_class, confidence = predict(image, model)

    st.markdown(f"### 🔭 Predicted Class: `{galaxy_class}`")
    st.markdown(f"**Confidence:** {confidence}%")

    with st.spinner("Generating explanation from knowledge base..."):
        explanation = get_answer(galaxy_class, vectorstore)

    st.markdown("### 📖 What is this galaxy?")
    st.write(explanation)