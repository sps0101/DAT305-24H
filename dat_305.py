# -*- coding: utf-8 -*-
"""

Task 6: Sentiment Analysis of US Airline Tweets

Goal: The goal of this project is to classify tweets related to US airlines into positive, neutral, or
negative sentiments. The students will design and implement a classification model to predict the
sentiment of airline-related tweets. They will experiment with various machine learning or deep
learning models, then evaluate their performance based on accuracy, precision, recall, and F1-
score. Students are encouraged to explore model optimization and fine-tuning techniques using
frameworks like Scikit-learn, TensorFlow, Keras, Transformer or PyTorch.
Dataset: The dataset is the US Airline Sentiment Dataset, containing 14,640 tweets labeled as
positive, neutral, or negative. The task is to predict the sentiment of each tweet, making it a
multiclass classification problem.

"""

#Importing libraries


import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter




#Loading the the dataset, clean the dateset, and also lemmatizer the words.

# Load the dataset
file_path = 'tweets.csv'  
tweets_df = pd.read_csv(file_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first rows before preprocessing
print("Before preprocessing:\n", tweets_df.columns)
print("Before preprocessing:\n", tweets_df.head())


# Lowercase the tweets
tweets_df['text'] = tweets_df['text'].str.lower()

# Remove punctuation marks
tweets_df['text'] = tweets_df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Tokenization, stop words removal, and stemming/lemmatization function
stop_words = set(stopwords.words('english'))

# Initialize PorterStemmer for stemming or WordNetLemmatizer for lemmatization
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


#Stemiing the tweets
def preprocess_text(text, use_stemming=True):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and keep only alphabetic tokens
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Apply stemming or lemmatization
    if use_stemming:
        processed_tokens = [ps.stem(word) for word in filtered_tokens]  # Stemming
    else:
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]  # Lemmatization
    
    # Join tokens back to string
    return ' '.join(processed_tokens)


# Apply preprocessing (use_stemming=True for stemming, False for lemmatization)
tweets_df['text'] = tweets_df['text'].apply(lambda x: preprocess_text(x, use_stemming=False))

# Display the first few rows after preprocessing
print("\nAfter preprocessing:\n", tweets_df['text'].head())

# Check basic information about the cleaned dataset
print(tweets_df.info())

# Check for missing values
print(tweets_df.isnull().sum())


#show how many categories are available in the dataset
unique_categories = tweets_df['airline_sentiment'].unique()
print("Unique sentiment categories:", unique_categories)

# To count the number of each category
category_counts = tweets_df['airline_sentiment'].value_counts()
print("Count of each sentiment category:\n", category_counts)



if 'airline_sentiment' in tweets_df.columns:
    sentiment_counts = tweets_df['airline_sentiment'].value_counts()
    unique_sentiments = tweets_df['airline_sentiment'].nunique()

    print(f"There are {unique_sentiments} unique sentiment categories:\n")
    print(sentiment_counts)
else:
    print("The 'airline_sentiment' column is not found in the dataset.")
    
    


#Plotting data, graph and word cloud

"""  
    
def plot_sentiment_distribution(df):
    sentiment_counts = df['airline_sentiment'].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Define function to plot word cloud for each sentiment category
def plot_word_cloud(df, sentiment):
    text = ' '.join(df[df['airline_sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.show()
    
    
plot_sentiment_distribution(tweets_df)

# Generate word clouds for each sentiment category
for sentiment in tweets_df['airline_sentiment'].unique():
    plot_word_cloud(tweets_df, sentiment)

# Plot for tid




def plot_sentiment_over_time(df):
    # Ensure 'tweet_created' is in datetime format
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])

    # Set 'tweet_created' as the index
    df.set_index('tweet_created', inplace=True)

    # Resample by day and count sentiments
    sentiment_over_time = df.groupby(['airline_sentiment']).resample('D').size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sentiment_over_time.plot(kind='line', marker='o')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y')
    plt.legend(title='Sentiment')
    plt.show()
    


# Call the function to plot sentiment over time
plot_sentiment_over_time(tweets_df)



def plot_tweets_per_day(df):
    if 'tweet_created' not in df.columns:
        print("Feil: Kolonnen 'tweet_created' finnes ikke.")
        return

    # Konverter 'tweet_created' til datetime
    try:
        df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
        df = df.dropna(subset=['tweet_created'])
    except Exception as e:
        print(f"Feil ved konvertering av 'tweet_created' til datetime: {e}")
        return
    
    # Sett 'tweet_created' som index
    df.set_index('tweet_created', inplace=True)

    # Tell tweets per dag
    tweets_per_day = df.resample('D').size()

    
    # Plot data
    plt.figure(figsize=(12, 6))
    tweets_per_day.plot(kind='bar', color='lightblue')
    plt.title('Number of Tweets Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Call the function to plot tweets per day
plot_tweets_per_day(tweets_df)




def plot_monthly_sentiment_trends(df):
    # Sjekk om nødvendige kolonner finnes
    required_columns = ['tweet_created', 'airline_sentiment']
    for column in required_columns:
        if column not in df.columns:
            print(f"Feil: Kolonnen '{column}' mangler i datasettet.")
            print("Tilgjengelige kolonner:", df.columns.tolist())
            return

    # Konverter 'tweet_created' til datetime-format, håndter ugyldige verdier
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    
    # Filtrer ut rader med ugyldige datoer
    df = df.dropna(subset=['tweet_created'])
    
    if df.empty:
        print("Datasettet inneholder ingen gyldige datoer etter filtrering.")
        return

    # Opprett en 'month_year'-kolonne for gruppering
    df['month_year'] = df['tweet_created'].dt.to_period('M')

    # Gruppér etter 'month_year' og 'airline_sentiment'
    try:
        monthly_sentiment = df.groupby(['month_year', 'airline_sentiment']).size().unstack(fill_value=0)
    except Exception as e:
        print(f"Feil ved gruppering av data: {e}")
        return

    # Plot data
    plt.figure(figsize=(12, 6))
    monthly_sentiment.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('Monthly Sentiment Trends')
    plt.xlabel('Month and Year')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

# Kall funksjonen for å plotte månedlige sentimenttrender
plot_monthly_sentiment_trends(tweets_df)


#Running deep learning model


# Step 1: Preprocessing - Tokenization and Encoding

# Build vocabulary and encode tweets
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(word_tokenize(text))
    vocab = {word: i + 1 for i, (word, _) in enumerate(counter.most_common())}  # index starts at 1
    vocab["<PAD>"] = 0  # padding token
    return vocab

def encode_text(text, vocab):
    return [vocab.get(word, vocab["<PAD>"]) for word in word_tokenize(text)]

# Building vocabulary and encoding text
vocab = build_vocab(tweets_df['text'])
tweets_df['encoded_text'] = tweets_df['text'].apply(lambda x: encode_text(x, vocab))

# Step 2: Dataset and DataLoader

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

# Map sentiment labels to integers (e.g., positive=2, neutral=1, negative=0)
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
tweets_df['label'] = tweets_df['airline_sentiment'].map(label_map)

# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    tweets_df['encoded_text'].tolist(),
    tweets_df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# DataLoader setup
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)
    return texts, labels

train_dataset = TweetDataset(train_texts, train_labels)
test_dataset = TweetDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Step 3: Model Architecture

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Instantiate model
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 64
output_dim = 3  # Three classes: negative, neutral, positive

model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim)

# Step 4: Training Loop

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    return predictions, true_labels

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

# Step 5: Evaluation
preds, true_labels = evaluate(model, test_loader)
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



print ("Do an extra model ")
print ("This is a Logistic Regression mode, anlysis of twees, transformed into TF-IDF features. ")



# Pandas display options to show all columns without width restriction
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first few rows to check data loading
print(tweets_df.head())

# Extract features (tweet text) and labels (sentiment)
X = tweets_df['text']  # Column with tweet content
y = tweets_df['airline_sentiment']  # Column with sentiment labels

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=100)  # Adjust max_iter if needed
log_reg.fit(X_train_tfidf, y_train)

# Predict the sentiments on the test set
y_pred = log_reg.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
"""
