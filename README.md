# CNN Digit Classifier — Streamlit App

Deploys the CNN trained in Notebook 1A as an interactive web app.

## Files
- `app.py` — complete Streamlit application source code
- `train_model.py` — recreates and saves the CNN (`digit_cnn_model.keras`) used by the app
- `digit_cnn_model.keras` — pre-trained model weights (already trained, ~98% test accuracy)
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (include `digit_cnn_model.keras`, `app.py`, `requirements.txt`).
2. Go to https://share.streamlit.io, connect the repo, and set the main file to `app.py`.
3. Deploy — no other configuration is required.

## Regenerating the model
If you want to retrain from scratch instead of using the bundled
`digit_cnn_model.keras`, run:
```bash
python train_model.py
```
This overwrites `digit_cnn_model.keras` with a freshly trained model.

## Notes
- The model was trained on 8x8 grayscale images (scikit-learn's `digits`
  dataset), so predictions work best on simple, bold, centered digit
  strokes — draw or upload accordingly.
- If `streamlit-drawable-canvas` isn't installed, the app automatically
  falls back to image-upload-only mode.
