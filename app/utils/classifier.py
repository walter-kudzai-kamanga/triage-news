# classifier implementation here (omitted for brevity)
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, List, Optional


class NewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_type = None

    def preprocess_text(self, text: str) -> str:
        """Preprocess text data"""
        if pd.isna(text) or text == "":
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Simple tokenization
        tokens = text.split()

        # Remove stopwords and short words
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        except:
            tokens = [word for word in tokens if len(word) > 2]

        # Stemming
        try:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        except:
            pass

        return ' '.join(tokens)

    def create_training_data(self) -> pd.DataFrame:
        """Create training dataset"""
        data = {
            'text': [
                # Politics
                "The government announced new economic policies today",
                "President signs new legislation into law",
                "Senate debates new healthcare bill in lengthy session",
                "Federal reserve raises interest rates to combat inflation",
                "Political party announces new leadership team",
                "Congress passes budget bill after weeks of negotiation",
                "Supreme court hears arguments on landmark case",
                "Prime minister meets with foreign diplomats",
                "New tax reform bill introduced in parliament",
                "Election results show clear victory for incumbent",

                # Sports
                "Football team wins championship after thrilling match",
                "Basketball star breaks scoring record in playoff game",
                "Olympic athlete sets world record in track event",
                "Soccer club signs multi-million dollar sponsorship deal",
                "Tennis player wins grand slam tournament",
                "Baseball pitcher throws perfect game in World Series",
                "Golf tournament ends with dramatic playoff victory",
                "Hockey team advances to finals with overtime win",
                "Swimming champion breaks personal best time",
                "Cycling competition sees new record set",

                # Business
                "Stock market reaches all-time high as companies report profits",
                "Corporate merger creates new market leader in tech industry",
                "Startup company secures venture capital funding",
                "Bank reports record quarterly earnings",
                "E-commerce giant expands into new markets",
                "Manufacturing company announces major expansion plans",
                "Tech company launches innovative new product",
                "Retail chain reports strong holiday sales",
                "Automotive industry sees surge in electric vehicle sales",
                "Real estate market shows signs of recovery"
            ],
            'category': [
                'politics', 'politics', 'politics', 'politics', 'politics',
                'politics', 'politics', 'politics', 'politics', 'politics',
                'sports', 'sports', 'sports', 'sports', 'sports',
                'sports', 'sports', 'sports', 'sports', 'sports',
                'business', 'business', 'business', 'business', 'business',
                'business', 'business', 'business', 'business', 'business'
            ]
        }
        return pd.DataFrame(data)

    def train(self, model_type: str = 'naive_bayes') -> Dict:
        """Train the classifier"""
        df = self.create_training_data()
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Encode labels
        y = self.label_encoder.fit_transform(df['category'])

        # Vectorize text
        X = self.vectorizer.fit_transform(df['processed_text']).toarray()

        # Train model
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError("Unsupported model type")

        self.model.fit(X, y)
        self.model_type = model_type

        # Calculate training accuracy
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return {
            "model_type": model_type,
            "accuracy": float(accuracy),
            "training_samples": len(df)
        }

    def predict(self, text: str) -> Dict:
        """Predict category for text"""
        if self.model is None:
            raise ValueError("Model not trained")

        processed_text = self.preprocess_text(text)
        X_new = self.vectorizer.transform([processed_text]).toarray()

        prediction = self.model.predict(X_new)[0]
        category = self.label_encoder.inverse_transform([prediction])[0]

        # Get probabilities if available
        probabilities = {}
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_new)[0]
            for i, cat in enumerate(self.label_encoder.classes_):
                probabilities[cat] = float(probs[i])

        return {
            "category": category,
            "probabilities": probabilities,
            "processed_text": processed_text
        }

    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model': self.model,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.model = model_data['model']
        self.model_type = model_data['model_type']


# Global classifier instance
classifier = NewsClassifier()