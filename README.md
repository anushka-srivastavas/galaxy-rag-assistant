# 🌌 Artemis Atlas

### AI-Powered Galaxy Morphology Classification & Astronomy Exploration Platform

Artemis Atlas is an end-to-end AI system that identifies galaxy morphologies from astronomical images and generates detailed scientific explanations using Retrieval-Augmented Generation (RAG).

Built at the intersection of Computer Vision, Large Language Models, and Astronomy, the platform enables users to upload a galaxy image, classify its morphology, visualize prediction confidence, and receive an AI-generated explanation describing its structure, formation, and notable real-world examples.

---

## 🚀 Live Demo

Try it here: https://artemisatlas.streamlit.app

---

## ✨ Features

- Classifies galaxies into 10 morphology categories using a fine-tuned ResNet-18 model.
- Displays prediction confidence and Top-3 class probabilities.
- Retrieves relevant astronomical knowledge using semantic search.
- Generates natural-language explanations using a Groq-hosted Llama model.
- Accessible through a public Streamlit web application.
- Supports real-time image upload and inference.

---

## 🔭 Galaxy Classes

- Disturbed Galaxy
- Merging Galaxy
- Round Smooth Galaxy
- In-between Round Smooth Galaxy
- Cigar Shaped Smooth Galaxy
- Barred Spiral Galaxy
- Unbarred Tight Spiral Galaxy
- Unbarred Loose Spiral Galaxy
- Edge-on Galaxy without Bulge
- Edge-on Galaxy with Bulge

---

## 🛠 Tech Stack

### Computer Vision
- PyTorch
- ResNet-18
- TorchVision

### Retrieval-Augmented Generation
- LangChain
- FAISS Vector Store
- HuggingFace Sentence Transformers

### Large Language Model
- Groq API
- Llama 3.3 70B Versatile

### Frontend & Deployment
- Streamlit
- Streamlit Cloud

### Dataset
- Galaxy10 DECals Dataset
- 17,736 galaxy images
- 10 morphology classes

---

## 🧠 System Architecture

Galaxy Image
      ↓
ResNet-18 Classifier
      ↓
Morphology Prediction
      ↓
Semantic Retrieval (FAISS)
      ↓
Astronomy Knowledge Context
      ↓
Groq Llama 3.3
      ↓
Natural Language Explanation

---

## 📊 Results

- Validation Accuracy: 84.3%
- Trained on 17,736 astronomical images
- Supports confidence-based interpretation through Top-3 predictions
- Successfully distinguishes major galaxy morphologies including spiral, elliptical, merging, and edge-on systems

---

artemis-atlas/
├── data/
├── models/
│   └── galaxy_classifier.pth
├── src/
│   ├── app.py
│   ├── classifier.py
│   └── rag.py
├── requirements.txt
└── README.md

---

## 💡 Motivation

As an astronomy enthusiast, I wanted to build a system that goes beyond image classification and helps users understand what they are observing. Artemis Atlas combines deep learning and retrieval-augmented generation to transform raw galaxy images into meaningful scientific insights, making astronomical exploration more accessible and interactive.