import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
#Load the dataset
file_path = r"C:\Users\Deepika\Desktop\PROJECTS\Fake-News-Detection\datasets\fake-news-detection(sheet-1).csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Display the first 5 rows
print("First few rows of the dataset:")
print(df.head())
# Check for null values
print("\nChecking for null values in the dataset:")
print(df.isnull().sum())
# Drop rows with missing values
df.dropna (inplace=True)
# Replace or remove invalid characters
df['Article code'] = df['Article code'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))
# Assuming the dataset has 'Article code' and 'Label' columns
X = df['Text'] # df['Article code']
y=df['Label']
# Check class balance
print("\nClass distribution:")
print(y.value_counts())
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
# Initialize the Logistic Regression model with class weights to handle imbalance 
model = LogisticRegression (class_weight='balanced')
# Train the model
model.fit(x_train_tfidf, y_train)
# Make predictions on the test set
y_pred = model.predict(x_test_tfidf)
# Evaluate the model
accuracy = accuracy_score (y_test, y_pred) 
print(f'\nModel Accuracy: {accuracy:.2f}')
# Display the classification report 
print("\nclassification Report:")
print(classification_report(y_test, y_pred))
# Print the size of the dataset 
print (f'\nDatasize: {len(X)}')
# Interactive part to enter an article code and get the prediction
def get_article_prediction (index):
  if 0 <= index < len(x_test):
    article = x_test.iloc[index]
    true_label = y_test.iloc[index]
    # Vectorize the article
    article_tfidf = vectorizer.transform([article])
    # Predict the label for the article
    predicted_label = model.predict(article_tfidf)[0]
    print(f"True Label: {'Fake' if true_label == 1 else 'Fake'}")
    print(f"\nArticle: {article}")
    print(f"Predicted Label: {'Fake' if predicted_label == 1 else 'Real'}")
  else:
       print(f"Invalid article code: {index}. Please enter a number between 0 and {len (x_test)-1}.")
# Loop to repeatedly ask for an article code until a valid one is entered
while True:
  try:
    article_code = int(input(f"Enter an article code (index) to check if it's fake or real (0 to {len(x_test) - 1}): ")) 
    get_article_prediction (article_code)
    break
  except ValueError:
    print("Please enter a valid integer.")
