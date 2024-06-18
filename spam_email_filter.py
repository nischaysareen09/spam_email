import pandas as pd

# Load the email dataset from the CSV file
emails = pd.read_csv(r'C:\Users\nisch\Documents\emails.csv')

# Print the first few rows and the column names to inspect the data
print(emails.head())
print(emails.columns)

import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization and stop word removal
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the email dataset from the CSV file
emails = pd.read_csv(r'C:\Users\nisch\Documents\emails.csv')

# Print the first few rows and the column names to inspect the data
print(emails.head())
print(emails.columns)

# Preprocess email texts
print("Preprocessing email texts...")
emails['text'] = emails['text'].apply(preprocess_text)
print("Preprocessing complete.")

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(emails['text'], emails['spam'], test_size=0.2, random_state=42)
print("Dataset split complete.")

# Create a pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes
print("Creating and training the pipeline...")
model = Pipeline([
    ('vectorizer', TfidfVectorizer(preprocessor=preprocess_text)),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)
print("Training complete.")

# Predict on the test set
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning using Grid Search
print("Starting hyperparameter tuning...")
param_grid = {
    'classifier__alpha': [0.1, 1.0],  # Reduced range for alpha
    'vectorizer__max_df': [0.75, 1.0],  # Reduced range for max_df
    'vectorizer__ngram_range': [(1, 1), (1, 2)]  # Same n-gram ranges
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2)  # Added verbosity for progress tracking
grid_search.fit(X_train, y_train)
print("Hyperparameter tuning complete.")

# Output best parameters from Grid Search
print("Best Parameters:", grid_search.best_params_)

# Retrain model with best parameters
print("Retraining model with best parameters...")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict on the test set with the best model
print("Making predictions with the best model on the test set...")
y_best_pred = best_model.predict(X_test)

# Evaluate the best model
print("Evaluating the best model...")
print("Accuracy with Best Model:", accuracy_score(y_test, y_best_pred))
print("Classification Report with Best Model:\n", classification_report(y_test, y_best_pred))

print("Script execution complete.")