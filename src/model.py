from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    vectorizer = TfidfVectorizer(max_features=5000)
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
