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
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import matplotlib.pyplot as plt
from wordcloud import WordCloud 

# Download necessary NLTK data (first-time use)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')  # For lemmatization support

# Load the dataset
file_path = 'tweets.csv'  # Replace with the correct file path
tweets_df = pd.read_csv(file_path)

# Display the first few rows before preprocessing
print("Before preprocessing:\n", tweets_df['text'].head())

# Lowercase the text
tweets_df['text'] = tweets_df['text'].str.lower()

# Remove punctuation marks
tweets_df['text'] = tweets_df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Tokenization, stop words removal, and stemming/lemmatization function
stop_words = set(stopwords.words('english'))

# Initialize PorterStemmer for stemming or WordNetLemmatizer for lemmatization
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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



if 'airline_sentiment' in tweets_df.columns:
    sentiment_counts = tweets_df['airline_sentiment'].value_counts()
    unique_sentiments = tweets_df['airline_sentiment'].nunique()

    print(f"There are {unique_sentiments} unique sentiment categories:\n")
    print(sentiment_counts)
else:
    print("The 'airline_sentiment' column is not found in the dataset.")
    
    
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
    # Ensure 'tweet_created' is in datetime format
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    
    # Set 'tweet_created' as the index
    df.set_index('tweet_created', inplace=True)

    # Count tweets per day
    tweets_per_day = df.resample('D').size()
    
    plt.figure(figsize=(12, 6))
    tweets_per_day.plot(kind='bar', color='lightblue')
    plt.title('Number of Tweets Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Call the function to plot tweets per day
plot_tweets_per_day(tweets_df)

def plot_monthly_sentiment_trends(df):
    # Ensure 'tweet_created' is in datetime format
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])

    # Create a 'month_year' column for grouping
    df['month_year'] = df['tweet_created'].dt.to_period('M')

    # Group by 'month_year' and 'airline_sentiment'
    monthly_sentiment = df.groupby(['month_year', 'airline_sentiment']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    monthly_sentiment.plot(kind='line', marker='o')
    plt.title('Monthly Sentiment Trends')
    plt.xlabel('Month and Year')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Sentiment')
    plt.show()

# Call the function to plot monthly sentiment trends
plot_monthly_sentiment_trends(tweets_df)


