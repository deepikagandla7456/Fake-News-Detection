import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Setup: Create a directory to save the models ---
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
    print("Created 'saved_models' directory.")

# --- Step 1: Data Loading and Preprocessing ---
print("Step 1: Loading and preprocessing data...")

# Load datasets
try:
    df_true = pd.read_csv("datasets/True.csv")
    df_false = pd.read_csv("datasets/Fake.csv")
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'True.csv' and 'False.csv' are in the 'datasets' directory.")
    exit()

# Add 'Label' column to each dataframe
df_true['Label'] = 'real'
df_false['Label'] = 'fake'

# Combine the two dataframes into one
df = pd.concat([df_true, df_false], ignore_index=True)

# Combine the 'title' and 'text' columns into a single feature column
# This gives the model more context for prediction
df['Text'] = df['title'] + " " + df['text']

# Drop the original columns that are no longer needed
df = df.drop(columns=['title', 'text', 'subject', 'date'])

# Shuffle the dataset to ensure random distribution
df = shuffle(df, random_state=42).reset_index(drop=True)

# Display dataset info
print("\n - Dataset Information -")
print(f"Total articles: {len(df)}")
print("Class distribution:\n", df['Label'].value_counts())
print("---------------------------\n")


# --- Step 2: Feature Engineering and Data Splitting ---
print("Step 2: Vectorizing text and splitting data...")

# Define features (X) and labels (y)
X_raw = df['Text'].astype(str) # Ensure all data is string type
y = df['Label']

# Vectorize the text data using TF-IDF
# We ignore common English stop words and terms that appear in more than 70% of documents
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X_raw)
print(" Text vectorization complete.")

# Split data into training and testing sets before resampling
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print(f" Data split into {len(y_train)} training samples and {len(y_test)} testing samples.")


# --- Step 3: Model Training ---
print("\nStep 3: Training the model...")

# Initialize and train the Logistic Regression model
model = LogisticRegression(solver='liblinear') # Using 'liblinear' is good for this dataset size
model.fit(X_train, y_train)
print("Model training complete")


# --- Step 4: Saving the Model and Vectorizer ---
print("\nStep 4: Saving model and vectorizer...")

# Save the trained model and the vectorizer to the 'saved_models' directory
joblib.dump(model, 'saved_models/model.joblib')
joblib.dump(vectorizer, 'saved_models/vectorizer.joblib')

print("\n--- TRAINING COMPLETED ---")
print("The trained model and vectorizer have been saved successfully.")



# --- Step 5: Evaluate the Model ---
print("\nStep 5: Evaluating model performance on the test set...")
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy Score: {accuracy:.4f}")

# Print the detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['fake', 'real']))

# Print Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion)

print("\n--- TRAINING AND EVALUATION COMPLETED ---")
print("The trained model and vectorizer have been saved successfully.")
