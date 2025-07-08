# Introvert or Extrovert Personality Classifier

## Overview
This project is an AI-powered text classification tool that predicts whether a person is an introvert or an extrovert based on a paragraph they provide. The model is trained on highly detailed, realistic data describing both personality types, and leverages advanced Natural Language Processing (NLP) and deep learning techniques.

## Features
- Classifies user input as "Introvert" or "Extrovert" based on a descriptive paragraph
- Uses a Bidirectional LSTM neural network for robust text understanding
- Advanced text preprocessing: stopword removal, lemmatization, and tokenization
- Model training includes early stopping, class weighting, and best model checkpointing
- Detailed evaluation with confusion matrix and classification report
- Easily extensible with more data or new categories

## Technologies Used
- **Python 3**
- **TensorFlow / Keras** (deep learning)
- **scikit-learn** (data splitting, evaluation, class weights)
- **NLTK** (stopwords, lemmatization)
- **NumPy** (data handling)

## Setup Instructions
1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Download NLTK data:**
   The script will automatically download required NLTK data (stopwords, wordnet) on first run.
4. **Prepare your data:**
   - Edit `data/introvert.txt` and `data/extrovert.txt` to further customize the training data if desired.

## How to Use
1. **Train and run the model:**
   ```bash
   python train_and_predict.py
   ```
   - The script will train the neural network, evaluate it, and prompt you to enter a paragraph describing yourself.
   - Enter a paragraph (likes, dislikes, habits, etc.) and the model will predict if you are likely an introvert or extrovert.
2. **Model files:**
   - The best model is saved as `best_personality_model.keras` and the tokenizer as `tokenizer.json`.
   - You can use these files for fast predictions in a separate script if desired.

## What is this project built on?
- **Technologies:** Python, TensorFlow/Keras, scikit-learn, NLTK, NumPy
- **AI Model:** Bidirectional LSTM neural network for text classification
- **NLP:** Advanced preprocessing (stopwords, lemmatization, tokenization)
- **Data:** Highly detailed, realistic descriptions of introvert and extrovert personalities

## Customization
- Add more data or categories by editing the `.txt` files in the `data/` folder.
- Tweak model parameters in `train_and_predict.py` for experimentation.

## Notes
- The more detailed and realistic your training data, the better the model will perform.
- For best results, use paragraphs that reflect real personality traits, preferences, and behaviors.

## Model Evaluation Example

After training, the script outputs a confusion matrix and a classification report to help you understand the model's performance. Here is a sample output:

```
              precision    recall  f1-score   support

   Introvert       0.00      0.00      0.00         4
   Extrovert       0.43      1.00      0.60         3

    accuracy                           0.43         7
   macro avg       0.21      0.50      0.30         7
weighted avg       0.18      0.43      0.26         7
```

- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1-score**: The harmonic mean of precision and recall (a balance between the two).
- **Support**: The number of true instances for each class in the test set.
- **Accuracy**: The overall proportion of correct predictions.
- **Macro avg**: The average of the metrics for each class, treating all classes equally.
- **Weighted avg**: The average of the metrics for each class, weighted by the number of true instances for each class.

These metrics help you evaluate how well the model is performing, especially if you add more data or adjust the model.

---

**requirements.txt** is included. Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

Enjoy exploring personality prediction with deep learning and NLP!