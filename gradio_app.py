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

# ─── Constants ────────────────────────────────────────────────────────────────
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

# ─── Load artifacts ───────────────────────────────────────────────────────────
model, tokenizer = None, None

def load_artifacts():
    global model, tokenizer
    if TF_AVAILABLE and os.path.exists("model.h5") and model is None:
        model = load_model("model.h5")
    if os.path.exists("tokenizer.pkl") and tokenizer is None:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

load_artifacts()

# ─── Preprocessing ────────────────────────────────────────────────────────────
def datacleaning(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)@[a-z0-9_]+", "", text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    text = [w for w in text.split() if w not in STOPWORDS]
    if NLTK_AVAILABLE:
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(w, 'v') for w in text]
    return ' '.join(text)

# ─── Predict function ─────────────────────────────────────────────────────────
def classify_news(article_text):
    if not article_text.strip():
        return {}, "Please enter some text."

    if model is None or tokenizer is None:
        return {}, "⚠️ Model not loaded. Ensure model.h5 and tokenizer.pkl exist."

    cleaned = datacleaning(article_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    probs = model.predict(padded, verbose=0)[0]

    # Return top-10 as label dict for Gradio Label component
    top_n = 10
    top_idx = np.argsort(probs)[::-1][:top_n]
    label_dict = {CATEGORIES[i]: float(probs[i]) for i in top_idx}

    top_cat = CATEGORIES[np.argmax(probs)]
    confidence = float(probs[np.argmax(probs)]) * 100
    summary = f"**Predicted Category:** {top_cat}  \n**Confidence:** {confidence:.1f}%\n\n**Cleaned text:** {cleaned}"

    return label_dict, summary

# ─── Example inputs ───────────────────────────────────────────────────────────
EXAMPLES = [
    ["NASA's James Webb telescope captures stunning images of distant galaxies formed just after the Big Bang."],
    ["Manchester United wins Premier League championship in dramatic final match comeback."],
    ["Apple unveils new AI-powered iPhone with groundbreaking neural processing chip."],
    ["Senate passes new climate bill after months of heated debate over energy regulations."],
    ["Top chefs reveal the secret behind making perfect sourdough bread at home."],
]

# ─── Gradio Interface ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="📰 News Article Classifier",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 900px !important; }"
) as demo:

    gr.Markdown("""
    # 📰 News Article Classifier
    ### Deep Learning · NLP · RNN/LSTM · 41 Categories
    Classify any news headline or article snippet into one of **41 news categories** 
    using a Bidirectional RNN model trained on the HuffPost News dataset.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="News Article / Headline",
                placeholder="Enter a news headline or article snippet here...",
                lines=5
            )
            classify_btn = gr.Button("🔍 Classify Article", variant="primary", size="lg")

        with gr.Column(scale=3):
            label_output = gr.Label(
                label="Top Predicted Categories",
                num_top_classes=5
            )
            info_output = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="💡 Try these examples"
    )

    with gr.Accordion("ℹ️ About this model", open=False):
        gr.Markdown(f"""
        **Model Architecture:** Embedding → Bidirectional SimpleRNN → Dense (Softmax)  
        **Dataset:** HuffPost News (~200k articles, 41 categories)  
        **Preprocessing:** Regex cleaning, stopword removal, WordNet lemmatization  
        **Tokenizer:** Keras Tokenizer with vocab size ~175,000  
        **Sequence length:** {MAXLEN} tokens (padded)  

        **All 41 categories:** {', '.join(CATEGORIES)}
        """)

    classify_btn.click(
        fn=classify_news,
        inputs=text_input,
        outputs=[label_output, info_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
