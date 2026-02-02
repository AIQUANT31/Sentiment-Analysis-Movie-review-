
import pandas as pd
import zipfile
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def load_local_dataset(zip_path):
    """Load dataset from local ZIP file"""
    print(f"Loading dataset from: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Files in ZIP: {zip_ref.namelist()}")
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        
        if csv_files:
            csv_name = csv_files[0]
            print(f"Extracting: {csv_name}")
            zip_ref.extract(csv_name)
            
            df = pd.read_csv(csv_name)
            return df
        else:
            raise FileNotFoundError("No CSV file found in the ZIP archive")

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance"""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report, y_pred

def predict_sentiment(model, vectorizer, text):
    """Predict sentiment for a single review"""
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    
    return "Positive" if prediction == 1 else "Negative"

def main():
    print("="*60)
    print("Sentiment Analysis for Movie Reviews")
    print("Using Naive Bayes and Logistic Regression")
    print("="*60)
    
    # Load local dataset
    zip_path = "/home/asharam-saini/Downloads/archive.zip"
    df = load_local_dataset(zip_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess text
    print("\nPreprocessing text data...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Map sentiment (handle different formats)
    if 'sentiment' in df.columns:
        # Convert to binary: positive=1, negative=0
        df['label'] = df['sentiment'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
    elif 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['pos', 'positive', '1', 1] else 0)
    
    X = df['cleaned_review'].tolist()
    y = df['label'].tolist()
    
    print(f"\nTotal reviews: {len(X)}")
    print(f"Positive reviews: {sum(y)}")
    print(f"Negative reviews: {len(y) - sum(y)}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train and evaluate Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression Model...")
    print("="*50)
    lr_model, lr_vectorizer = train_logistic_regression(X_train, y_train)
    lr_accuracy, lr_report, lr_predictions = evaluate_model(
        lr_model, lr_vectorizer, X_test, y_test
    )
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print("\nClassification Report:")
    print(lr_report)
    
    # Train and evaluate Naive Bayes
    print("\n" + "="*50)
    print("Training Naive Bayes Model...")
    print("="*50)
    nb_model, nb_vectorizer = train_naive_bayes(X_train, y_train)
    nb_accuracy, nb_report, nb_predictions = evaluate_model(
        nb_model, nb_vectorizer, X_test, y_test
    )
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    print("\nClassification Report:")
    print(nb_report)
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"Logistic Regression: {lr_accuracy:.4f}")
    print(f"Naive Bayes: {nb_accuracy:.4f}")
    
    if lr_accuracy > nb_accuracy:
        print(f"\nLogistic Regression performs better by {(lr_accuracy - nb_accuracy):.4f}")
    elif nb_accuracy > lr_accuracy:
        print(f"\nNaive Bayes performs better by {(nb_accuracy - lr_accuracy):.4f}")
    else:
        print("\nBoth models have equal performance")
    
    # Example predictions
    print("\n" + "="*50)
    print("Example Predictions")
    print("="*50)
    
    sample_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible movie, wasted my time. Complete waste of money.",
        "I really enjoyed this film. The cinematography was beautiful.",
        "Boring and predictable. I fell asleep during the movie."
    ]
    
    print("\nUsing Logistic Regression:")
    for review in sample_reviews:
        sentiment = predict_sentiment(lr_model, lr_vectorizer, review)
        print(f"Review: '{review[:50]}...' -> {sentiment}")
    
    print("\nUsing Naive Bayes:")
    for review in sample_reviews:
        sentiment = predict_sentiment(nb_model, nb_vectorizer, review)
        print(f"Review: '{review[:50]}...' -> {sentiment}")

if __name__ == "__main__":
    main()
