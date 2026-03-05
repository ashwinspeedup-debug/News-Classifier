import streamlit as st
import numpy as np
import re
import pickle
import os

# Try importing optional libraries
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

# ─── Constants ───────────────────────────────────────────────────────────────
MAXLEN = 130
NUM_CLASSES = 41
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

CATEGORY_EMOJI = {
    'POLITICS': '🏛️', 'SPORTS': '⚽', 'TECH': '💻', 'ENTERTAINMENT': '🎬',
    'BUSINESS': '💼', 'CRIME': '🚨', 'SCIENCE': '🔬', 'TRAVEL': '✈️',
    'FOOD & DRINK': '🍽️', 'WELLNESS': '🧘', 'EDUCATION': '📚',
    'ENVIRONMENT': '🌿', 'COMEDY': '😄', 'RELIGION': '🙏', 'MEDIA': '📺',
    'MONEY': '💰', 'STYLE': '👗', 'STYLE & BEAUTY': '💄', 'PARENTING': '👨‍👩‍👧',
    'DIVORCE': '💔', 'WOMEN': '👩', 'ARTS': '🎨', 'ARTS & CULTURE': '🎭',
    'HOME & LIVING': '🏠', 'HEALTHY LIVING': '💪', 'WEIRD NEWS': '🤪',
    'WEDDINGS': '💒', 'GREEN': '♻️', 'IMPACT': '🌍', 'TASTE': '😋',
    'COLLEGE': '🎓', 'FIFTY': '🎂', 'GOOD NEWS': '🌟', 'QUEER VOICES': '🏳️‍🌈',
    'LATINO VOICES': '🌮', 'BLACK VOICES': '✊', 'THE WORLDPOST': '🌐',
    'U.S. NEWS': '🇺🇸', 'WORLD NEWS': '🌎', 'CULTURE & ARTS': '🎭',
    'PARENTS': '👨‍👩‍👦'
}

# ─── Text Preprocessing ───────────────────────────────────────────────────────
def datacleaning(text):
    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(' ', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    text = [word for word in text.split() if word not in STOPWORDS]
    if NLTK_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        sentence = [lemmatizer.lemmatize(word, 'v') for word in text]
    else:
        sentence = text
    return ' '.join(sentence)


@st.cache_resource
def load_artifacts():
    """
    Load model and tokenizer.
    If model is not present locally, download it first.
    """

    import gdown
    model, tokenizer = None, None

    MODEL_PATH = "model.h5"
    TOKENIZER_PATH = "tokenizer.pkl"

    # Google Drive file ID
    MODEL_URL = "https://drive.google.com/file/d/1fRsSUPsxshZX3qAXN1k7oxJpoVFvljSL/view?usp=sharing"

    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False,fuzzy=True)

    # Load model
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)

    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    return model, tokenizer


def predict(text, model, tokenizer):
    cleaned = datacleaning(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    probs = model.predict(padded, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    results = [(CATEGORIES[i], float(probs[i]) * 100) for i in top3_idx]
    return results, cleaned


# ─── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; text-align: center; }
    .subtitle   { text-align: center; color: #666; margin-bottom: 2rem; }
    .result-card {
        background: #f0f4ff; border-radius: 12px;
        padding: 1rem 1.5rem; margin: 0.5rem 0;
        border-left: 5px solid #4f6ef7;
    }
    .top-result  { border-left: 5px solid #22c55e; background: #f0fff4; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📰 News Article Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep Learning · NLP · RNN/LSTM · 41 Categories</p>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This model classifies news articles into **41 categories** using:
    - **Keras Tokenizer** for text vectorization
    - **Bidirectional SimpleRNN** + **Embedding Layer**
    - Trained on the HuffPost News dataset (~200k articles)
    """)
    st.markdown("---")
    st.write("**Categories include:**")
    for cat in sorted(CATEGORIES):
        emoji = CATEGORY_EMOJI.get(cat, '📌')
        st.write(f"{emoji} {cat}")

# Main input
st.subheader("Enter a news headline or article snippet")
input_text = st.text_area(
    label="",
    placeholder="e.g. 'Scientists discover new exoplanet in habitable zone...'",
    height=150
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Classify", use_container_width=True)

if predict_btn:
    if not input_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        model, tokenizer = load_artifacts()
        if model is None or tokenizer is None:
            st.error("⚠️ Model or tokenizer not found. Please make sure `model.h5` and `tokenizer.pkl` are in the app directory.")
            st.info("**Quick start:** Save your trained Keras model as `model.h5` and your tokenizer as `tokenizer.pkl` using pickle, then place them alongside this app file.")
        else:
            with st.spinner("Classifying..."):
                results, cleaned = predict(input_text, model, tokenizer)

            st.markdown("### 🏆 Prediction Results")
            for i, (cat, conf) in enumerate(results):
                emoji = CATEGORY_EMOJI.get(cat, '📌')
                card_class = "result-card top-result" if i == 0 else "result-card"
                label = "✅ Top Prediction" if i == 0 else f"#{i+1}"
                st.markdown(f"""
                <div class="{card_class}">
                    <strong>{label}: {emoji} {cat}</strong><br>
                    Confidence: <b>{conf:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf / 100)

            with st.expander("🔍 Preprocessed text"):
                st.code(cleaned)

# Demo examples
st.markdown("---")
st.subheader("💡 Try an example")
examples = {
    "🔬 Science": "NASA's James Webb telescope captures stunning images of distant galaxies formed just after the Big Bang.",
    "⚽ Sports": "Manchester United wins the Premier League championship after dramatic comeback in final match.",
    "💻 Tech": "Apple unveils new AI-powered iPhone with groundbreaking neural processing chip at annual event.",
    "🏛️ Politics": "Senate passes new climate bill after months of heated debate over energy regulations.",
    "🍽️ Food": "Top chefs reveal the secret behind making perfect sourdough bread at home with simple techniques.",
}
selected = st.selectbox("Pick an example article", list(examples.keys()))
if st.button("Load Example"):
    st.session_state["example_text"] = examples[selected]
    st.rerun()

if "example_text" in st.session_state:
    st.text_area("Example loaded — copy and paste above:", value=st.session_state["example_text"], height=100)
