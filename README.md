# 📰 News Article Classifier — Deployment Guide

Bidirectional RNN · NLP · Keras · 41 Categories · HuffPost Dataset

---

## 📁 File Structure

```
news_classifier_deployment/
├── app.py                # Hugging Face Spaces (Gradio)
├── gradio_app.py         # Standalone Gradio app
├── streamlit_app.py      # Streamlit app
├── save_artifacts.py     # Script to export model + tokenizer
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🔧 Step 1 — Export Model Artifacts from Your Notebook

At the end of your training notebook, run:

```python
import pickle

# Save the trained Keras model
model.save("model.h5")

# Save the Keras Tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

Then **copy `model.h5` and `tokenizer.pkl` into this folder**.

---

## 🚀 Deployment Options

---

### 1️⃣ Streamlit (Local / Streamlit Cloud)

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Deploy to Streamlit Cloud:**
1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to `streamlit_app.py`
4. Add `model.h5` and `tokenizer.pkl` to the repo (or use Git LFS for large files)
5. Click **Deploy** ✅

> ⚠️ If `model.h5` is >100MB, use [Git LFS](https://git-lfs.github.com/):
> ```bash
> git lfs install
> git lfs track "*.h5"
> git add .gitattributes model.h5
> ```

---

### 2️⃣ Gradio (Local / Standalone)

```bash
pip install gradio tensorflow nltk
python gradio_app.py
```

A local URL will be printed. Add `share=True` in `demo.launch()` to get a public link.

---

### 3️⃣ Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   - **SDK:** Gradio
   - **Python:** 3.9+

2. Upload these files to the Space repo:
   ```
   app.py
   requirements.txt
   model.h5
   tokenizer.pkl
   ```

3. For large model files (>50MB), use the HF Hub CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload <your-username>/<space-name> model.h5
   huggingface-cli upload <your-username>/<space-name> tokenizer.pkl
   ```

4. HF Spaces will auto-build and deploy from `app.py` ✅

**Example `README.md` header for HF Spaces** (add to top of this file):
```yaml
---
title: News Article Classifier
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
---
```

---

## 🧪 Quick Test (No Model Needed)

The apps will show a clear error message if `model.h5` / `tokenizer.pkl` are missing, so you can verify the UI works before adding model files.

---

## 📊 Model Summary

| Attribute | Value |
|---|---|
| Architecture | Embedding → BiRNN → Dense (Softmax) |
| Dataset | HuffPost News (~185k articles) |
| Output classes | 41 categories |
| Sequence length | 130 tokens |
| Best test accuracy | ~49% (SimpleRNN baseline) |

---

## 💡 Tips to Improve Accuracy

- Replace SimpleRNN with **LSTM** or **GRU** layers
- Use **pre-trained embeddings** (GloVe, FastText)
- Try **transformer-based models** (DistilBERT, BERT) via HuggingFace
- Increase `maxlen` to 200+ tokens
