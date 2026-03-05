"""
Hugging Face Spaces entry point — identical to gradio_app.py
but with HF-specific launch settings.
"""
import gradio as gr
import numpy as np
import re
import pickle
import os

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = set()

MAXLEN = 130
CATEGORIES = [
    'ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE',
    'COMEDY', 'CRIME', 'CULTURE & ARTS', 'DIVORCE', 'EDUCATION',
    'ENTERTAINMENT', 'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK', 'GOOD NEWS',
    'GREEN', 'HEALTHY LIVING', 'HOME & LIVING', 'IMPACT', 'LATINO VOICES',
    'MEDIA', 'MONEY', 'PARENTING', 'PARENTS', 'POLITICS', 'QUEER VOICES',
    'RELIGION', 'SCIENCE', 'SPORTS', 'STYLE', 'STYLE & BEAUTY', 'TASTE',
    'TECH', 'THE WORLDPOST', 'TRAVEL', 'U.S. NEWS', 'WEDDINGS',
    'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS'
]

model, tokenizer = None, None

def load_artifacts():
    global model, tokenizer
    if TF_AVAILABLE and os.path.exists("model.h5") and model is None:
        try:
            model = load_model("model.h5")
            print("✅ Model loaded.")
        except Exception as e:
            print(f"❌ Model load error: {e}")
    if os.path.exists("tokenizer.pkl") and tokenizer is None:
        try:
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded.")
        except Exception as e:
            print(f"❌ Tokenizer load error: {e}")

load_artifacts()

def datacleaning(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)@[a-z0-9_]+", "", text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    if NLTK_AVAILABLE:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(w, 'v') for w in tokens]
    return ' '.join(tokens)

def classify_news(article_text):
    if not article_text.strip():
        return {cat: 0.0 for cat in CATEGORIES[:5]}, "⚠️ Please enter some text."
    if model is None or tokenizer is None:
        return {cat: 0.0 for cat in CATEGORIES[:5]}, "⚠️ Model files not found. Upload `model.h5` and `tokenizer.pkl`."
    cleaned = datacleaning(article_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    probs = model.predict(padded, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:10]
    label_dict = {CATEGORIES[i]: float(probs[i]) for i in top_idx}
    top_cat = CATEGORIES[int(np.argmax(probs))]
    conf = float(probs[np.argmax(probs)]) * 100
    summary = f"**Top Category:** {top_cat} ({conf:.1f}% confidence)\n\n**Preprocessed:** `{cleaned[:200]}...`"
    return label_dict, summary

EXAMPLES = [
    ["NASA's James Webb telescope captures stunning images of galaxies formed just after the Big Bang."],
    ["Senate passes sweeping new climate legislation amid growing concerns over global temperatures."],
    ["Apple unveils new AI-powered chip in its latest iPhone lineup at annual September event."],
    ["Manchester United scores late winner to clinch dramatic Premier League title on final day."],
    ["Nutritionists reveal the top superfoods to boost immunity during the winter months."],
]

with gr.Blocks(title="📰 News Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📰 News Article Classifier
    **Bidirectional RNN · NLP · 41 Categories · HuffPost Dataset**
    
    Enter any news headline or snippet to classify it into one of 41 news categories.
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="📝 Article / Headline",
                placeholder="Enter a news headline or article snippet...",
                lines=4
            )
            btn = gr.Button("Classify →", variant="primary")

        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="🏆 Predictions")
            out_md = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=inp)

    with gr.Accordion("📖 Model Info", open=False):
        gr.Markdown(f"""
        | Attribute | Detail |
        |---|---|
        | Architecture | Embedding → Bidirectional SimpleRNN → Softmax |
        | Dataset | HuffPost News (~185k articles) |
        | Classes | 41 news categories |
        | Sequence length | {MAXLEN} tokens |
        | Preprocessing | Regex + Stopwords + Lemmatization |
        """)

    btn.click(fn=classify_news, inputs=inp, outputs=[out_label, out_md])

demo.launch()
