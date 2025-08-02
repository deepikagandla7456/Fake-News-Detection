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
