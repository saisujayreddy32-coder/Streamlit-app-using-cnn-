# 🔢 CNN Digit Classifier — Streamlit App

A Streamlit web application that deploys a Convolutional Neural Network (CNN)
to classify handwritten digits (0–9) in real time. Draw a digit on the
in-browser canvas or upload an image, and the app predicts the digit along
with a confidence score and full class-probability breakdown.

> Model architecture, training, and evaluation details are documented in
> the companion notebook: `1A_CNN_Algorithm_Notebook.ipynb`.

## Features
- ✏️ **Draw or upload** — sketch a digit on an interactive canvas, or upload a PNG/JPG image
- ⚡ **Real-time inference** — predictions update instantly using a pre-trained Keras model
- 📊 **Confidence breakdown** — bar chart of predicted probabilities across all 10 digit classes
- 🛡️ **Graceful fallbacks** — clear error messaging if the model file is missing, and automatic fallback to upload-only mode if the drawing component isn't installed

## Tech Stack
`Streamlit` · `TensorFlow / Keras` · `NumPy` · `Pillow` · `Pandas`

## Project Structure
```
.
├── app.py                    # Streamlit application source code
├── train_model.py            # Script to (re)train and save the CNN
├── digit_cnn_model.keras     # Pre-trained model weights (~98% test accuracy)
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting Started

### Run locally
```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`.

### Retrain the model (optional)
```bash
python train_model.py
```
This regenerates `digit_cnn_model.keras` from scratch.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (make sure `digit_cnn_model.keras` is included).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file path to `app.py` and deploy — no extra configuration needed.

## Notes
- The underlying model was trained on 8×8 grayscale images (scikit-learn's
  `digits` dataset), so predictions work best on simple, bold, centered
  digit strokes.
- Test accuracy: **~98%**

## License
Add your preferred license here (e.g., MIT).
