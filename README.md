# 🧠 MindLift – AI Emotional Support Quote Assistant

> A beginner-friendly deep learning project that detects your emotional mood from text and responds with a personalised motivational quote.

---

## 📌 Project Overview

**MindLift** takes a sentence written by the user, runs it through a trained LSTM-based deep learning model, and:

1. Predicts the **mood** (Happy 😊 / Sad 😢 / Angry 😠 / Neutral 😐)
2. Shows a **confidence score** for the prediction
3. Returns a hand-picked **motivational quote** matched to the detected mood

---

## 🗂️ Project Structure

```
mindlift/
│
├── dataset.csv        ← 163 labelled text samples (4 moods)
├── train_model.py     ← Builds, trains, and saves the DL model
├── app.py             ← Streamlit web app (UI)
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
│
│   (generated after training)
├── model.h5           ← Saved Keras model
└── tokenizer.pkl      ← Saved tokenizer + label encoder
```

---

## 🏗️ Model Architecture

```
Input text
    │
    ▼
Tokenizer  →  integer sequence  →  Padding (max_len=30)
    │
    ▼
Embedding Layer   (vocab=3000, dim=64)
    │
    ▼
SpatialDropout1D  (rate=0.2)
    │
    ▼
LSTM Layer        (units=64, dropout=0.2)
    │
    ▼
Dense Layer       (32 units, ReLU)
    │
    ▼
Dropout           (rate=0.3)
    │
    ▼
Dense Output      (4 units, Softmax)  →  [happy, sad, angry, neutral]
```

---

## ⚙️ Setup Instructions

### 1 · Clone / Download the project

```bash
# If downloaded as a zip, unzip it, then:
cd mindlift
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation may take a few minutes.

---

## 🚀 Running the Project

### Step 1 – Train the model

```bash
python train_model.py
```

**Expected output:**
```
=======================================================
  MindLift – Model Training Started
=======================================================

[1/6] Loading dataset …
      Total samples : 163
      Mood counts   :
      sad        44
      happy      44
      angry      41
      neutral    34

[2/6] Encoding labels …
      Classes : ['angry', 'happy', 'neutral', 'sad']

[3/6] Tokenizing and padding text …
      Vocabulary size : 393
      Input shape     : (163, 30)

[4/6] Building model …
      ...

[5/6] Training …
      Epoch 1/30 ...
      Epoch 12/30 ...  ← EarlyStopping kicks in

  ✓ Best validation accuracy : 91.7 %

[6/6] Saving model and tokenizer …
  ✓ model.h5 saved
  ✓ tokenizer.pkl saved

=======================================================
  Training complete!  Run:  streamlit run app.py
=======================================================
```

### Step 2 – Launch the web app

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501**

---

## 💡 Example Outputs

| Input text | Predicted Mood | Confidence |
|---|---|---|
| "I feel absolutely wonderful today" | 😊 Happy | 97 % |
| "I have been crying all day and nothing helps" | 😢 Sad | 94 % |
| "I am furious about how I was treated" | 😠 Angry | 96 % |
| "Today was a regular day at work" | 😐 Neutral | 88 % |

**Sample quote (Sad mood):**
> *"Every storm runs out of rain. Brighter days are on their way. 🌈"*

**Sample quote (Happy mood):**
> *"Your joy is contagious — keep spreading it! 🌟"*

---

## 🧪 Key Concepts Used

| Concept | Where Used |
|---|---|
| **Embedding Layer** | Converts word indices to dense vectors |
| **LSTM** | Learns sequential/temporal patterns in text |
| **Tokenizer** | Maps words to integer indices |
| **Padding** | Makes all sequences the same fixed length |
| **Label Encoding** | Converts mood strings to numbers |
| **One-hot Encoding** | Converts labels for categorical crossentropy |
| **EarlyStopping** | Prevents overfitting by stopping training early |
| **Softmax** | Outputs probabilities over all 4 mood classes |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Deep learning framework (Keras inside) |
| `streamlit` | Web UI framework |
| `scikit-learn` | Train/test split, label encoding |
| `pandas` | Dataset loading |
| `numpy` | Numerical operations |

---

## 👨‍💻 Author Notes

This project is designed to be completed in a single day and is suitable for a university assignment submission. All code is heavily commented for easy understanding.

Feel free to extend it by:
- Adding more mood categories (e.g. anxious, excited)
- Using a pre-trained embedding like GloVe
- Deploying the Streamlit app to Streamlit Cloud

---

*MindLift v1.0 · Deep Learning Mini Project*
