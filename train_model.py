# ============================================================
# MindLift – AI Emotional Support Quote Assistant
# train_model.py  |  Deep Learning Text Classification
# ============================================================
# FIXES APPLIED (v2):
#   - class_weight balancing so no mood is ignored
#   - Bidirectional LSTM for better context understanding
#   - Larger embedding + LSTM units
#   - ReduceLROnPlateau to escape stuck training
#   - Stratified split so all moods appear in validation
#   - Sanity check after training
# ============================================================

import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
VOCAB_SIZE  = 5000
MAX_LEN     = 40
EMBED_DIM   = 128
LSTM_UNITS  = 128
BATCH_SIZE  = 8
EPOCHS      = 60
TEST_SIZE   = 0.15

print("=" * 55)
print("  MindLift – Model Training v2 (Bias Fixed)")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset ...")
df = pd.read_csv("dataset.csv")
print(f"      Samples : {len(df)}")
print(f"      Distribution:\n{df['mood'].value_counts().to_string()}\n")

texts  = df["text"].astype(str).tolist()
labels = df["mood"].tolist()

# ─────────────────────────────────────────────────────────
# ENCODE LABELS
# ─────────────────────────────────────────────────────────
print("[2/6] Encoding labels ...")
label_encoder  = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes    = len(label_encoder.classes_)
print(f"      Classes : {list(label_encoder.classes_)}")

y = to_categorical(encoded_labels, num_classes=num_classes)

# KEY FIX 1: class_weight — forces equal attention to ALL moods
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(encoded_labels),
    y=encoded_labels
)
class_weight_dict = dict(enumerate(class_weights_array))
print(f"      Class weights : { {label_encoder.classes_[k]: round(v,2) for k,v in class_weight_dict.items()} }")

# ─────────────────────────────────────────────────────────
# TOKENIZE & PAD
# ─────────────────────────────────────────────────────────
print("\n[3/6] Tokenizing and padding ...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
print(f"      Vocab size  : {len(tokenizer.word_index)}")
print(f"      Input shape : {X.shape}")

# ─────────────────────────────────────────────────────────
# STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, encoded_labels,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=encoded_labels
)
print(f"\n      Train : {len(X_train)}  |  Val : {len(X_val)}")

# ─────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────
print("\n[4/6] Building model ...")

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.15),
    # KEY FIX 2: Bidirectional LSTM reads sentence both ways
    # catches patterns like "not happy" vs "happy"
    Bidirectional(LSTM(LSTM_UNITS, dropout=0.15, recurrent_dropout=0.15)),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ─────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    # KEY FIX 3: halves LR when training is stuck
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
]

# ─────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────
print("\n[5/6] Training ...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,   # KEY FIX: penalise missed moods
    callbacks=callbacks,
    verbose=1,
)

best_val_acc = max(history.history["val_accuracy"])
print(f"\n  Best validation accuracy : {best_val_acc * 100:.1f} %")

# ─────────────────────────────────────────────────────────
# SANITY CHECK — one sentence per mood
# ─────────────────────────────────────────────────────────
print("\n  Sanity check:")
tests = {
    "happy"  : "I feel absolutely wonderful and joyful today",
    "sad"    : "I feel so hopeless and nothing is going right",
    "angry"  : "I am furious and cannot control my rage",
    "neutral": "Today was a regular and ordinary day at work",
}
for expected, sentence in tests.items():
    seq    = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs  = model.predict(padded, verbose=0)[0]
    pred   = label_encoder.inverse_transform([np.argmax(probs)])[0]
    conf   = float(np.max(probs)) * 100
    mark   = "OK" if pred == expected else "MISS"
    print(f"  [{mark}]  expected={expected:<7}  got={pred:<7}  {conf:.1f}%")

# ─────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────
print("\n[6/6] Saving ...")
model.save("model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer,
                 "label_encoder": label_encoder,
                 "max_len": MAX_LEN}, f)
print("  model.h5 saved")
print("  tokenizer.pkl saved")
print("\n" + "="*55)
print("  Done! Run:  streamlit run app.py")
print("="*55)
