# 🌌 Galaxy Morphology Classifier with RAG Explanation Pipeline

A multimodal AI system that classifies galaxy morphologies from images and generates detailed explanations using a Retrieval-Augmented Generation (RAG) pipeline.

## Demo
Upload any galaxy image → ResNet-18 classifies the morphology → RAG pipeline retrieves relevant astronomy knowledge → Mistral LLM generates a detailed explanation.

## Tech Stack
- **Classification**: ResNet-18 (PyTorch) fine-tuned on Galaxy10 DECals dataset
- **Dataset**: Galaxy10 DECals — 17,736 galaxy images across 10 morphology classes
- **RAG Pipeline**: LangChain + FAISS + HuggingFace sentence-transformers
- **LLM**: Mistral via Ollama (local inference)
- **Frontend**: Streamlit

## Project Structure
```
galaxy-rag-assistant/
├── models/              # Trained ResNet-18 weights
├── src/
│   ├── classifier.py    # Image classification module
│   ├── rag.py           # RAG pipeline with FAISS vectorstore
│   └── app.py           # Streamlit frontend
└── requirements.txt
```

## Galaxy Classes
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

## Results
- **Validation Accuracy**: 84.3% across 10 classes
- **Training**: 10 epochs with data augmentation and learning rate scheduling
- Correctly classifies standard galaxy morphologies with high confidence
- Appropriately expresses low confidence on edge cases outside training distribution (e.g. ring galaxies)

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/anushka-srivastavas/galaxy-rag-assistant
cd galaxy-rag-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama with Mistral
```bash
ollama pull mistral
```

### 4. Add the trained model
Place `galaxy_classifier.pth` in the `models/` folder.

### 5. Run the app
```bash
streamlit run src/app.py
```