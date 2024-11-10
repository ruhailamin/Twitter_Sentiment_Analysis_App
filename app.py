import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('wordnet')  # Downloading the WordNet resource for lemmatization
nltk.download('punkt')     # Downloading punkt for tokenization

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Set of English stop words
stop_words = set(stopwords.words('english'))

# Expanded list of negation words and contractions
negation_patterns = [
    r"\b(can't|cannot|can not)\b",       # can't, cannot, can not
    r"\b(won't|will not)\b",             # won't, will not
    r"\b(doesn't|does not)\b",          # doesn't, does not
    r"\b(isn't|is not)\b",              # isn't, is not
    r"\b(aren't|are not)\b",            # aren't, are not
    r"\b(weren't|were not)\b",          # weren't, were not
    r"\b(shouldn't|should not)\b",      # shouldn't, should not
    r"\b(wouldn't|would not)\b",        # wouldn't, would not
    r"\b(couldn't|could not)\b",        # couldn't, could not
    r"\b(no|not)\b"                     # no, not
]

# Function to detect negation
def handle_negation(text):
    # Check if any negation word exists in the text
    return any(re.search(pattern, text) for pattern in negation_patterns)

# Updated Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and special characters, keeping only letters, numbers, spaces, and apostrophes
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization and Lemmatization function
def tokenize_and_lemmatize(text):
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    # Remove stop words and apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Join the lemmatized tokens back into a string
    return ' '.join(lemmatized_tokens)

# Streamlit UI setup
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = load('sentiment_model.sav')
vocab = load('tfidf_vocab.joblib')
vectorizer = TfidfVectorizer(vocabulary=vocab)

st.markdown('<h1  style = "color:blue" class = "custom_title">Sentiment Analysis of Tweets</h1>', unsafe_allow_html=True)
user_tweet = st.text_area("Enter your tweet:")

if st.button("Predict Sentiment"):
    if user_tweet:
        # Preprocess the tweet
        processed_tweet = preprocess_text(user_tweet)
        # Tokenize and lemmatize
        processed_tweet = tokenize_and_lemmatize(processed_tweet)
        # Vectorize the processed tweet
        vectorized_tweet = vectorizer.fit_transform([processed_tweet])
        # Predict sentiment
        sentiment = model.predict(vectorized_tweet)[0]
        
        # Detect if the tweet contains negation and adjust sentiment
        negation_found = handle_negation(user_tweet)
        if negation_found:
            sentiment = 1 - sentiment  # Invert sentiment if negation is found
        
        sentiment_label = "Positive" if sentiment == 1 else "Negative"
        # Display sentiment result
        st.write(f"The sentiment of this tweet is: **{sentiment_label}**")
    else:
        st.warning("Please enter a tweet to analyze.")
