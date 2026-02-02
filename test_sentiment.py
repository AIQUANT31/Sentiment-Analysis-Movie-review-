"""
Test sentiment analysis with your own input
Usage: python3 test_sentiment.py
Then enter your movie reviews interactively!
Type 'quit', 'exit', or 'q' to exit.
"""
from load_model import load_lr_model, predict

def main():
    print("="*60)
    print("Sentiment Analysis - Test Your Review (Logistic Regression)")
    print("="*60)
    
    # Load Logistic Regression model (better accuracy & precision)
    print("Loading Logistic Regression model...")
    lr_model, lr_vectorizer = load_lr_model()
    print("Model loaded successfully!\n")
    
    print("Enter your movie review to get sentiment prediction.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")
    print("-"*60)
    
    # While loop for continuous user input
    while True:
        try:
            # Get user input
            review = input("Enter review: ").strip()
            
            # Check if user wants to exit
            if review.lower() in ['quit', 'exit', 'q']:
                print("\n" + "="*60)
                print("Thank you for using Sentiment Analysis!")
                print("Goodbye!")
                print("="*60)
                break
            
            # Skip empty input
            if not review:
                print("Please enter a review or type 'q' to quit.\n")
                continue
            
            # Get prediction using Logistic Regression
            sentiment = predict(lr_model, lr_vectorizer, review)
            
            # Display result
            print(f"Review: {review[:100]}{'...' if len(review) > 100 else ''}")
            print(f"Prediction: {sentiment}")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()

