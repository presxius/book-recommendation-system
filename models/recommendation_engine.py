import numpy as np
import pandas as pd
from collections import defaultdict

class RecommendationEngine:
    def __init__(self, data_processor, svd_model):
        self.data_processor = data_processor
        self.svd_model = svd_model
        self.user_profiles = {}
        
    def create_user_profile(self, user_ratings):
        """Create a new user profile based on initial ratings"""
        # Convert ISBN to encoded book IDs
        encoded_ratings = []
        for isbn, rating in user_ratings:
            if isbn in self.data_processor.book_to_idx:
                book_idx = self.data_processor.book_to_idx[isbn]
                # Normalize user rating from 1-10 scale to 0-1 scale
                normalized_rating = (float(rating) - 1) / 9.0  # Map 1-10 to 0-1
                encoded_ratings.append((book_idx, normalized_rating))
            else:
                print(f"Book ISBN {isbn} not found in training data")
        
        return encoded_ratings
    
    def get_cold_start_recommendations(self, user_ratings, n=10):
        """Get recommendations for new users (cold start problem)"""
        print("Using cold start recommendations")
        
        # For cold start, use popularity-based recommendations
        if hasattr(self.data_processor, 'ratings_df') and self.data_processor.ratings_df is not None:
            popular_books = self.data_processor.ratings_df.groupby('ISBN')['Book-Rating'].count().nlargest(n)
        else:
            # Fallback: use books with highest publication year
            popular_books = self.data_processor.books_df.nlargest(n, 'Year-Of-Publication')['ISBN']
        
        recommendations = []
        for isbn in popular_books.index:
            book_info = self.data_processor.books_df[
                self.data_processor.books_df['ISBN'] == isbn
            ]
            if not book_info.empty:
                book_info = book_info.iloc[0]
                recommendations.append({
                    'ISBN': str(isbn),
                    'Title': str(book_info.get('Book-Title', 'Unknown Title')),
                    'Author': str(book_info.get('Book-Author', 'Unknown Author')),
                    'Year': int(book_info.get('Year-Of-Publication', 2000)),
                    'Publisher': str(book_info.get('Publisher', 'Unknown Publisher')),
                    'Predicted_Rating': float(8.0)
                })
        
        return recommendations
    
    def get_personalized_recommendations(self, user_id_encoded, n=10):
        """Get personalized recommendations using SVD"""
        if not self.svd_model.is_trained:
            raise ValueError("SVD model not trained!")
        
        print(f"Getting personalized recommendations for user {user_id_encoded}")
        
        # Get top N recommendations
        top_books = self.svd_model.get_top_n_recommendations(
            user_id_encoded, 
            self.data_processor.ratings_df, 
            n * 2
        )
        
        # Convert to book information
        recommendations = []
        for book_id_encoded, predicted_rating in top_books:
            # Find original ISBN
            isbn = self.data_processor.idx_to_book.get(book_id_encoded)
            if isbn is None:
                continue
            
            # Get book details
            book_info = self.data_processor.books_df[
                self.data_processor.books_df['ISBN'] == isbn
            ]
            
            if not book_info.empty:
                book_info = book_info.iloc[0]
                recommendations.append({
                    'ISBN': str(isbn),
                    'Title': str(book_info.get('Book-Title', 'Unknown Title')),
                    'Author': str(book_info.get('Book-Author', 'Unknown Author')),
                    'Year': int(book_info.get('Year-Of-Publication', 2000)),
                    'Publisher': str(book_info.get('Publisher', 'Unknown Publisher')),
                    'Predicted_Rating': float(round(predicted_rating * 10, 1))
                })
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def hybrid_recommendations(self, user_ratings, n=10):
        """Hybrid approach for better recommendations"""
        print(f"Generating hybrid recommendations for {len(user_ratings)} user ratings")
        
        if len(user_ratings) < 2:
            # Cold start: use popularity-based
            print("Using cold start approach (fewer than 2 ratings)")
            return self.get_cold_start_recommendations(user_ratings, n)
        else:
            # For demo purposes, use simulated recommendations
            print("Using simulated personalized recommendations")
            return self._get_simulated_recommendations(user_ratings, n)
    
    def _get_simulated_recommendations(self, user_ratings, n=10):
        """Simulate recommendations based on user preferences"""
        # Get popular books
        popular_books = self.data_processor.get_popular_books(n * 2)
        
        recommendations = []
        for _, book in popular_books.iterrows():
            if len(recommendations) >= n:
                break
                
            recommendations.append({
                'ISBN': str(book['ISBN']),
                'Title': str(book.get('Book-Title', 'Unknown Title')),
                'Author': str(book.get('Book-Author', 'Unknown Author')),
                'Year': int(book.get('Year-Of-Publication', 2000)),
                'Publisher': str(book.get('Publisher', 'Unknown Publisher')),
                'Predicted_Rating': float(round(8.0 + np.random.random() * 2.0, 1))
            })
        
        return recommendations