# ============================================================
# MindLift – AI Emotional Support Quote Assistant
# app.py  |  Streamlit Web Application
# ============================================================
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pickle
import random
import os

# ── TensorFlow (suppress noisy logs) ─────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindLift – Emotional Support",
    page_icon="🧠",
    layout="centered",
)

# ─────────────────────────────────────────────────────────
# MOTIVATIONAL QUOTES  (keyed by mood)
# ─────────────────────────────────────────────────────────
QUOTES = {
    "happy": [
        "Your joy is contagious — keep spreading it! 🌟",
        "Happiness is not something ready-made; it comes from your own actions. – Dalai Lama",
        "You are radiating positive energy. Keep shining! ✨",
        "The secret of joy is to keep seeking something greater than yourself. – Sarah Ban Breathnach",
        "Your smile is your superpower. Never stop using it! 😊",
        "Joy is not in things; it is in us. – Richard Wagner",
        "Today you are you — that is truer than true. – Dr. Seuss",
    ],
    "sad": [
        "Every storm runs out of rain. Brighter days are on their way. 🌈",
        "You are stronger than you think, and this pain will not last forever. 💪",
        "It's okay to feel sad — your feelings are valid. Be gentle with yourself. 🫂",
        "Stars can't shine without darkness. Your light is still there. ⭐",
        "Even the darkest night will end, and the sun will rise. – Victor Hugo",
        "Tears water the seeds of tomorrow's strength. 🌱",
        "Healing is not linear — every small step forward counts. 💙",
    ],
    "angry": [
        "Take a deep breath. You have the strength to respond, not react. 🌬️",
        "Speak when you are angry and you'll make the best speech you'll ever regret. – Ambrose Bierce",
        "Your anger is valid, but don't let it steal your peace. 🕊️",
        "Between stimulus and response there is space. In that space is your power. – Viktor Frankl",
        "Breathe. Pause. The situation does not have power over you unless you give it. 🧘",
        "You are in control. Channel this energy into something powerful. ⚡",
        "Peace does not mean no conflict; it means handling conflict peacefully. ☮️",
    ],
    "neutral": [
        "Every ordinary day is a gift. Make the most of it! 🎁",
        "Small, consistent actions lead to extraordinary results. Keep going. 🚀",
        "Life is not waiting for the storm to pass — it's learning to dance in the rain. 💃",
        "You don't have to feel great to do great things. Start anyway. 🛤️",
        "Routine is the foundation that extraordinary things are built upon. 🏗️",
        "Even calm waters can carry great ships. 🚢",
        "Embrace the ordinary — it holds the seeds of something special. 🌿",
    ],
}

# Emoji badge per mood
MOOD_META = {
    "happy"  : {"emoji": "😊", "color": "#F6C90E", "label": "Happy",   "bg": "#FFF9E6"},
    "sad"    : {"emoji": "😢", "color": "#5B9BD5", "label": "Sad",     "bg": "#EBF3FB"},
    "angry"  : {"emoji": "😠", "color": "#E74C3C", "label": "Angry",   "bg": "#FDEDEC"},
    "neutral": {"emoji": "😐", "color": "#2ECC71", "label": "Neutral", "bg": "#EAFAF1"},
}

# ─────────────────────────────────────────────────────────
# LOAD MODEL & TOKENIZER  (cached so they load only once)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load the trained model and tokenizer from disk."""
    if not os.path.exists("model.h5") or not os.path.exists("tokenizer.pkl"):
        return None, None, None, None

    model = tf.keras.models.load_model("model.h5")

    with open("tokenizer.pkl", "rb") as f:
        bundle = pickle.load(f)

    tokenizer     = bundle["tokenizer"]
    label_encoder = bundle["label_encoder"]
    max_len       = bundle["max_len"]

    return model, tokenizer, label_encoder, max_len


model, tokenizer, label_encoder, MAX_LEN = load_assets()

# ─────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────
def predict_mood(text: str):
    """
    Given a user sentence, returns:
      - predicted mood (str)
      - confidence score (float 0-1)
      - probabilities for all classes (dict)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenize and pad the input text
    seq     = tokenizer.texts_to_sequences([text])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Model prediction → array of probabilities
    probs   = model.predict(padded, verbose=0)[0]

    # Highest probability index → mood label
    pred_idx    = int(np.argmax(probs))
    mood        = label_encoder.inverse_transform([pred_idx])[0]
    confidence  = float(probs[pred_idx])

    # Build a readable probability dict for all moods
    all_probs = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probs)
    }

    return mood, confidence, all_probs


# ─────────────────────────────────────────────────────────
# CUSTOM CSS  – clean, calming design
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fdf6f0 100%);
}

/* ── Header ── */
.mindlift-header {
    text-align: center;
    padding: 2rem 0 0.5rem;
}
.mindlift-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1a1a2e;
    margin: 0;
    letter-spacing: -0.5px;
}
.mindlift-header p {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Input card ── */
.input-card {
    background: white;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin: 1.5rem 0;
}

/* ── Result card ── */
.result-card {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
}

/* ── Mood badge ── */
.mood-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* ── Quote box ── */
.quote-box {
    border-left: 4px solid;
    padding: 0.9rem 1.2rem;
    border-radius: 0 12px 12px 0;
    font-size: 1.05rem;
    font-style: italic;
    line-height: 1.7;
    margin-top: 1rem;
    background: rgba(255,255,255,0.6);
}

/* ── Confidence bar label ── */
.conf-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #9ca3af;
    margin-bottom: 0.15rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #d1d5db;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
}

/* Streamlit button override */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="mindlift-header">
    <h1>🧠 MindLift</h1>
    <p>Your AI-powered emotional support companion — tell me how you feel.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CHECK IF MODEL IS LOADED
# ─────────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️  Model not found!\n\n"
        "Please run training first:\n"
        "```\npython train_model.py\n```\n"
        "Then relaunch the app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("##### 💬 How are you feeling right now?")

user_input = st.text_area(
    label="",
    placeholder="e.g. I feel really down today and nothing seems to be going right…",
    height=110,
    key="user_text",
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyse = st.button("✨  Analyse My Mood", use_container_width=True)
with col_clear:
    if st.button("🔄  Clear", use_container_width=True):
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PREDICTION & RESULTS
# ─────────────────────────────────────────────────────────
if analyse:
    text = user_input.strip()

    if len(text) < 3:
        st.warning("Please enter at least a few words so I can understand your mood. 🙏")
    else:
        with st.spinner("Reading your emotions …"):
            mood, confidence, all_probs = predict_mood(text)

        meta  = MOOD_META[mood]
        quote = random.choice(QUOTES[mood])

        # ── Result card ───────────────────────────────
        st.markdown(
            f"""
            <div class="result-card" style="background:{meta['bg']}; border: 1.5px solid {meta['color']}30;">
                <span class="mood-badge" style="background:{meta['color']}22; color:{meta['color']};">
                    {meta['emoji']} {meta['label']}
                </span>
                <div class="conf-label">Confidence: {confidence*100:.1f}%</div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar (native Streamlit progress)
        st.progress(confidence)

        # Quote
        st.markdown(
            f"""
            <div class="quote-box" style="border-color:{meta['color']}; color:#374151;">
                {quote}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Mood probability breakdown (expander) ─────
        with st.expander("📊 Full mood probability breakdown"):
            for m, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                m_meta = MOOD_META[m]
                st.markdown(
                    f"**{m_meta['emoji']} {m_meta['label']}** — {prob*100:.1f}%"
                )
                st.progress(prob)

# ─────────────────────────────────────────────────────────
# SIDEBAR – About
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 About MindLift")
    st.markdown(
        "MindLift uses a **deep learning** model "
        "(Embedding → LSTM → Dense) to classify your text "
        "into one of four moods and respond with a "
        "personalised motivational quote.\n\n"
        "**Detectable moods:**\n"
        "- 😊 Happy\n- 😢 Sad\n- 😠 Angry\n- 😐 Neutral"
    )
    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        "- Python 3.x\n"
        "- TensorFlow / Keras\n"
        "- Streamlit\n"
        "- scikit-learn\n"
        "- pandas / numpy"
    )
    st.divider()
    st.caption("MindLift v1.0 · University Project · 2024")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Made with ❤️ · MindLift – AI Emotional Support Quote Assistant</div>',
    unsafe_allow_html=True,
)
