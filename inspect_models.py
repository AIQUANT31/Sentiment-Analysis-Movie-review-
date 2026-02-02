"""
Inspect saved model files
"""
import pickle

def inspect_pickle_file(filepath):
    """Load and inspect a pickle file"""
    print(f"\n{'='*50}")
    print(f"Inspecting: {filepath}")
    print('='*50)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        print(f"Content: {data}")
        
        # If it's a model, show more details
        if hasattr(data, 'predict'):
            print("This is a sklearn model with predict method")
        if hasattr(data, 'transform'):
            print("This is a transformer (vectorizer)")
            
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Inspect all model files
    model_files = [
        'models/lr_model.pkl',
        'models/lr_vectorizer.pkl',
        'models/nb_model.pkl',
        'models/nb_vectorizer.pkl'
    ]
    
    for filepath in model_files:
        inspect_pickle_file(filepath)
