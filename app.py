<<<<<<< HEAD
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
=======
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv("Enter your file path")

# Display basic info
print("First few rows of the dataset:\n", df.head(), "\n")
print("Checking for null values in the dataset:\n", df.isnull().sum(), "\n")
print("Class distribution:\n", df['Label'].value_counts(), "\n")
print("Datasize:", len(df))

# Feature and label
X_raw = df['Text']
y = df['Label']

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_raw)

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- User Prediction Interface ---
while True:
    user_input = input(f"\nEnter an article index (0 to {len(df)-1}) to check if it's fake or real (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting program. Thank you!")
        break

    try:
        index = int(user_input)
        if 0 <= index < len(df):
            article = df.iloc[index]['Text']
            true_label = df.iloc[index]['Label']
            vectorized_article = vectorizer.transform([article])
            prediction = model.predict(vectorized_article)[0]

            print("\n Article:\n", article)
            print("\n True Label:", true_label.capitalize())
            print(" Predicted Label:", prediction.capitalize())
        else:
            print(f" Please enter a number between 0 and {len(df)-1}.")
    except ValueError:
        print(" Invalid input. Enter a valid index number or type 'exit' to quit.")
>>>>>>> b9af10e42afd10bda0d5b21b125f6e050c9811b8
