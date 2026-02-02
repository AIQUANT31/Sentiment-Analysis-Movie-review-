"""
Sentiment Analysis for Movie Reviews
Using Naive Bayes and Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_imdb_reviews
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text
from src.model import (
    train_logistic_regression,
    train_naive_bayes,
    evaluate_model
)

def main():
    # Load IMDB movie reviews dataset
    print("Loading IMDB movie reviews dataset...")
    imdb = fetch_imdb_reviews()
    X = imdb.data
    y = imdb.target
    
    # Convert bytes to string and preprocess
    print("Preprocessing text data...")
    X = [clean_text(review.decode('utf-8')) for review in X]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
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

if __name__ == "__main__":
    main()
