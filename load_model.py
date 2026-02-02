"""
Load saved models and make predictions
"""

import pickle
from src.preprocess import clean_text

def load_lr_model():
    """Load Logistic Regression model"""
    with open('models/lr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/lr_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def load_nb_model():
    """Load Naive Bayes model"""
    with open('models/nb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/nb_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict(model, vectorizer, text):
    """Predict sentiment for a review"""
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    # Load models
    print("Loading models...")
    lr_model, lr_vectorizer = load_lr_model()
    nb_model, nb_vectorizer = load_nb_model()
    print("Models loaded successfully!")
    
    # Test predictions
    test_reviews = [
        "This movie was absolutely fantastic!",
        "Terrible movie, complete waste of time.",
        "I really enjoyed the acting and storyline.",
        "Boring and predictable."
    ]
    
    print("\n" + "="*50)
    print("Predictions using Logistic Regression:")
    print("="*50)
    for review in test_reviews:
        sentiment = predict(lr_model, lr_vectorizer, review)
        print(f"'{review}' -> {sentiment}")
    
    print("\n" + "="*50)
    print("Predictions using Naive Bayes:")
    print("="*50)
    for review in test_reviews:
        sentiment = predict(nb_model, nb_vectorizer, review)
        print(f"'{review}' -> {sentiment}")
