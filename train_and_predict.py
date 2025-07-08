import os
import re
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not present
nltk.download('stopwords')
nltk.download('wordnet')

# Parameters
INTROVERT_FILE = 'data/introvert.txt'
EXTROVERT_FILE = 'data/extrovert.txt'
MAX_NUM_WORDS = 3000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 128
MODEL_PATH = 'personality_model.h5'
BEST_MODEL_PATH = 'best_personality_model.keras'
TOKENIZER_PATH = 'tokenizer.json'

# Advanced text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, stop_words, lemmatizer):
    text = clean_text(text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Load data
def load_data():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    texts = []
    labels = []
    with open(INTROVERT_FILE, encoding='utf-8') as f:
        for para in f.read().split('\n\n'):
            para = para.strip()
            if para:
                texts.append(preprocess_text(para, stop_words, lemmatizer))
                labels.append(0)
    with open(EXTROVERT_FILE, encoding='utf-8') as f:
        for para in f.read().split('\n\n'):
            para = para.strip()
            if para:
                texts.append(preprocess_text(para, stop_words, lemmatizer))
                labels.append(1)
    return texts, labels

texts, labels = load_data()
labels = np.array(labels)

# Tokenize
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Save tokenizer as JSON
with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Compute class weights for imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Improved model: Bidirectional LSTM + BatchNorm + Dropout
def build_model():
    model = Sequential([
        Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Callbacks for early stopping and best model saving
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

# Train
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=8,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# Save final model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Best model saved to {BEST_MODEL_PATH}")
print(f"Tokenizer saved to {TOKENIZER_PATH}")

# Evaluation
print("\nEvaluation on test set:")
y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))

# Prediction function using saved model and tokenizer
def predict_paragraph_with_saved_model(paragraph):
    # Load tokenizer from JSON
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_json_str = f.read()
    loaded_tokenizer = tokenizer_from_json(tokenizer_json_str)
    # Preprocess input
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = preprocess_text(paragraph, stop_words, lemmatizer)
    seq = loaded_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    loaded_model = load_model(BEST_MODEL_PATH)
    pred = loaded_model.predict(padded)
    label = np.argmax(pred, axis=1)[0]
    return 'Introvert' if label == 0 else 'Extrovert'

# User input
if __name__ == '__main__':
    print("\nEnter a paragraph describing yourself (likes, dislikes, habits, etc.):")
    user_input = input()
    result = predict_paragraph_with_saved_model(user_input)
    print(f"\nPrediction: You are likely an {result}.") 