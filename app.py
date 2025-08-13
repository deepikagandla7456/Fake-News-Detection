import joblib
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- Step 1: Load the Trained Model and Vectorizer ---
try:
    # Load the model and vectorizer from the saved_models directory
    model = joblib.load('saved_models/model.joblib')
    vectorizer = joblib.load('saved_models/vectorizer.joblib')
    print(" Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("\n Error: Model or vectorizer not found.")
    print("Please run 'train_model.py' first to train and save the necessary files.")
    exit()

# --- Step 2: Prediction Function ---
def predict_news(text, model_to_use, vectorizer_to_use):
    """
    Predicts if a given news text is 'real' or 'fake' and returns the confidence.

    Args:
        text (str): The news article text to classify.
        model_to_use: The trained classification model.
        vectorizer_to_use: The fitted TF-IDF vectorizer.

    Returns:
        tuple: (The predicted label, the confidence score).
    """
    # Use the passed-in vectorizer to transform the input text
    vectorized_text = vectorizer_to_use.transform([text])
    
    # Use the passed-in model to make a prediction
    prediction = model_to_use.predict(vectorized_text)[0]
    
    # Get the prediction probabilities
    confidence_scores = model_to_use.predict_proba(vectorized_text)
    
    # Find the confidence of the predicted class
    prediction_confidence = confidence_scores.max() * 100
    
    return prediction, prediction_confidence

# --- Step 3: Main Application Loop ---
if __name__ == "__main__":
    print("\n--- Fake News Detection ---")
    print("Enter a news article text to check its authenticity.")
    print("Type 'exit' or 'quit' to close the program.\n")

    while True:
        # Prompt the user for input
        user_input = input("Enter news text here: ")

        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Add a check for very short or empty input
        if len(user_input.strip()) < 20:
            print("\n Please enter a more complete article text for an accurate prediction.\n")
            continue

        # Get the prediction and confidence by passing the model and vectorizer
        prediction_label, confidence = predict_news(user_input, model, vectorizer)

        # Display the result to the user
        print("\n---------------------------------")
        print(f"PREDICTION: This news is likely {prediction_label.upper()}.")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("---------------------------------\n")