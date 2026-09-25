# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import unicodedata
import regex as re

def fix_mojibake(text):
    """Perbaiki karakter mojibake dari latin-1 ke UTF-8."""
    try:
        return text.encode("latin1").decode("utf-8")
    except:
        return text

class DataPreprocessor:
    def __init__(self):
        self.books_df = None
        self.users_df = None
        self.ratings_df = None
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        
    def load_data(self, books_path, users_path, ratings_path):
        """Load datasets from CSV files"""
        try:
            self.books_df = pd.read_csv(books_path, encoding='latin-1', low_memory=False)
            self.users_df = pd.read_csv(users_path, encoding='latin-1')
            self.ratings_df = pd.read_csv(ratings_path, encoding='latin-1')
            print("Data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_string(self, text):
        if pd.isna(text):
            return ""

        text = str(text)

        # Perbaiki mojibake
        text = fix_mojibake(text)

        # Normalisasi unicode global
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\p{L}\p{N}\p{M}\s\.\,\-\!\?\'\"]+", "", text)

        text = " ".join(text.split())

        return text.strip()
    
    def preprocess_books(self):
        """Preprocess books data"""
        print("Preprocessing books data...")
        
        # Handle missing values
        self.books_df['Book-Author'] = self.books_df['Book-Author'].fillna('Unknown')
        self.books_df['Publisher'] = self.books_df['Publisher'].fillna('Unknown')
        self.books_df['Year-Of-Publication'] = pd.to_numeric(
            self.books_df['Year-Of-Publication'], errors='coerce'
        ).fillna(2000).astype(int)
        
        # Clean text fields
        self.books_df['Book-Title'] = self.books_df['Book-Title'].apply(self.clean_string)
        self.books_df['Book-Author'] = self.books_df['Book-Author'].apply(self.clean_string)
        self.books_df['Publisher'] = self.books_df['Publisher'].apply(self.clean_string)
        
        # Remove duplicates
        self.books_df = self.books_df.drop_duplicates(subset=['ISBN'])
        
        print(f"Books data preprocessed: {len(self.books_df)} records")
        return self.books_df
    
    def preprocess_users(self):
        """Preprocess users data"""
        print("Preprocessing users data...")
        
        # Handle missing values
        self.users_df['Age'] = self.users_df['Age'].fillna(self.users_df['Age'].median())
        self.users_df['Location'] = self.users_df['Location'].fillna('Unknown')
        
        # Clean location data
        self.users_df['Location'] = self.users_df['Location'].apply(self.clean_string)
        
        # Remove duplicates
        self.users_df = self.users_df.drop_duplicates(subset=['User-ID'])
        
        print(f"Users data preprocessed: {len(self.users_df)} records")
        return self.users_df
    
    def preprocess_ratings(self):
        """Preprocess ratings data dengan implicit feedback handling dan normalisasi"""
        print("Preprocessing ratings data...")
        
        # Remove duplicates
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['User-ID', 'ISBN'])
        
        # Filter ratings to only include users and books that exist in respective datasets
        valid_users = set(self.users_df['User-ID']) if self.users_df is not None else set(self.ratings_df['User-ID'])
        valid_books = set(self.books_df['ISBN']) if self.books_df is not None else set(self.ratings_df['ISBN'])
        
        self.ratings_df = self.ratings_df[
            self.ratings_df['User-ID'].isin(valid_users) & 
            self.ratings_df['ISBN'].isin(valid_books)
        ]
        
        # IMPLICIT FEEDBACK HANDLING: Treat rating 0 as implicit interaction
        print("Handling implicit feedback...")
        
        # Pisahkan explicit ratings (1-10) dan implicit feedback (0)
        explicit_ratings = self.ratings_df[self.ratings_df['Book-Rating'] > 0].copy()
        implicit_interactions = self.ratings_df[self.ratings_df['Book-Rating'] == 0].copy()
        
        self.ratings_df = explicit_ratings
        
        print(f"Explicit ratings: {len(explicit_ratings)}")
        print(f"Implicit interactions (excluded from training): {len(implicit_interactions)}")
        print(f"Final ratings for training: {len(self.ratings_df)}")
        
        # NORMALISASI: Convert scale 1-10 to 1-5
        print("Normalizing ratings scale from 1-10 to 1-5...")
        scaler = MinMaxScaler(feature_range=(1, 5))
        
        # Convert ratings to numpy array dan reshape
        ratings_values = self.ratings_df['Book-Rating'].values.reshape(-1, 1)
        normalized_ratings = scaler.fit_transform(ratings_values)
        
        # Update ratings dengan nilai normalized
        self.ratings_df['Book-Rating'] = normalized_ratings.flatten()
        
        print(f"Ratings normalized: {np.min(normalized_ratings):.2f} to {np.max(normalized_ratings):.2f}")
        
        # Encode user and book IDs for model training
        self.ratings_df['user_id_encoded'] = self.user_encoder.fit_transform(self.ratings_df['User-ID'])
        self.ratings_df['book_id_encoded'] = self.book_encoder.fit_transform(self.ratings_df['ISBN'])
        
        print(f"Ratings data preprocessed: {len(self.ratings_df)} records")
        return self.ratings_df
    
    def get_preprocessed_data(self):
        """Return all preprocessed data"""
        return {
            'books': self.books_df,
            'users': self.users_df,
            'ratings': self.ratings_df,
            'user_encoder': self.user_encoder,
            'book_encoder': self.book_encoder
        }
    
    def get_data_statistics(self):
        """Return data statistics for display"""
        stats = {
            'books_count': len(self.books_df) if self.books_df is not None else 0,
            'users_count': len(self.users_df) if self.users_df is not None else 0,
            'ratings_count': len(self.ratings_df) if self.ratings_df is not None else 0
        }
        
        if self.ratings_df is not None:
            stats.update({
                'min_rating': float(self.ratings_df['Book-Rating'].min()),
                'max_rating': float(self.ratings_df['Book-Rating'].max()),
                'avg_rating': float(self.ratings_df['Book-Rating'].mean()),
                'rating_std': float(self.ratings_df['Book-Rating'].std())
            })
        
        return stats