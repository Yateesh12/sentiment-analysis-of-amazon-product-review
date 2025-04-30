# STEP 1: Install required libraries
!pip install -q nltk scikit-learn joblib

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords

nltk.download('stopwords')

# Use Python’s built-in CSV reader to detect bad lines and clean them
clean_lines = []
with open('Reviews.csv', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            # Check for expected number of columns (assume 10 for Amazon dataset, you can adjust)
            if len(row) >= 10:
                clean_lines.append(row)
        except:
            continue

# Create DataFrame manually
columns = clean_lines[0]
data = clean_lines[1:]
df = pd.DataFrame(data, columns=columns)
df = df[['Score', 'Summary']]  # Adjust if necessary

# STEP 4: Convert 'Score' column to int
df['Score'] = df['Score'].astype(int)

# Define sentiment labeling function
def convert_rating(Score):
    if Score >= 4:
        return 'positive'
    elif Score == 3:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment labeling
df['sentiment'] = df['Score'].apply(convert_rating)

# STEP 5: Clean text data
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['cleaned_summary'] = df['Summary'].apply(clean_text)

# STEP 6: Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_summary'])
y = df['sentiment']

# STEP 7: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 8: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 9: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# STEP 10: Visualizations

# 1. Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'neutral', 'negative'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 3. Accuracy Bar Chart
accuracy = accuracy_score(y_test, y_pred)
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy Score')
plt.tight_layout()
plt.show()

# STEP 11: Save model (optional)
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print(df.columns.tolist())
