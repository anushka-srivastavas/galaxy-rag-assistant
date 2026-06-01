# 🌌 Galaxy Morphology Classifier with RAG-Powered Astronomy Explanations

An end-to-end AI system that classifies galaxy morphologies from astronomical images and generates detailed natural-language explanations using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

Web App: https://artemisatlas.streamlit.app

Upload a galaxy image and receive:

- Morphology classification
- Prediction confidence
- Top-3 class probabilities
- AI-generated astronomy explanation

---

## ✨ Features

- Classifies galaxies into 10 morphology categories using a fine-tuned ResNet-18 model
- Generates astronomy-focused explanations using a RAG pipeline
- Retrieves relevant galaxy information from a FAISS vector database
- Uses Groq-hosted Llama 3.3 for fast cloud inference
- Accessible through a public Streamlit web application

---

## 🛠 Tech Stack

### Computer Vision
- PyTorch
- ResNet-18
- Galaxy10 DECals Dataset

### Retrieval-Augmented Generation
- LangChain
- FAISS Vector Store
- HuggingFace Sentence Transformers

### Large Language Model
- Groq API
- Llama 3.3 70B Versatile

### Frontend
- Streamlit

---

## 📂 Project Structure

```text
galaxy-rag-assistant/
├── data/
├── models/
│   └── galaxy_classifier.pth
├── src/
│   ├── app.py
│   ├── classifier.py
│   └── rag.py
├── requirements.txt
└── README.md
```

---

## 🔭 Galaxy Classes

1. Disturbed Galaxy
2. Merging Galaxy
3. Round Smooth Galaxy
4. In-between Round Smooth Galaxy
5. Cigar Shaped Smooth Galaxy
6. Barred Spiral Galaxy
7. Unbarred Tight Spiral Galaxy
8. Unbarred Loose Spiral Galaxy
9. Edge-on Galaxy without Bulge
10. Edge-on Galaxy with Bulge

---

## 📊 Model Performance

- Validation Accuracy: 84.3%
- Trained on 17,736 galaxy images from Galaxy10 DECals
- Data augmentation and learning-rate scheduling applied during training
- Supports confidence-based interpretation through Top-3 predictions

---

## 🧠 System Architecture

text Galaxy Image       ↓ ResNet-18 Classifier       ↓ Predicted Morphology       ↓ FAISS Retrieval       ↓ Relevant Astronomy Context       ↓ Groq Llama 3.3       ↓ Natural Language Explanation 

---

## 🔬 Dataset

Dataset: Galaxy10 DECals

- 17,736 galaxy images
- 10 morphology classes
- Derived from the Sloan Digital Sky Survey (SDSS)

---

## 💡 Motivation

As an astronomy enthusiast, I wanted to combine computer vision and large language models to create an interactive system that not only identifies galaxy morphologies but also explains the science behind them in a way that is accessible and engaging.