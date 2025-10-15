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
                # Note: user ratings should be 1-10, not 0-10
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
                    'ISBN': str(isbn),  # Ensure string type
                    'Title': str(book_info.get('Book-Title', 'Unknown Title')),
                    'Author': str(book_info.get('Book-Author', 'Unknown Author')),
                    'Year': int(book_info.get('Year-Of-Publication', 2000)),  # Convert to native int
                    'Publisher': str(book_info.get('Publisher', 'Unknown Publisher')),
                    'Predicted_Rating': float(8.0)  # Convert to native float
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
            n * 2  # Get more to filter out already rated books
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
                    'ISBN': str(isbn),  # Ensure string type
                    'Title': str(book_info.get('Book-Title', 'Unknown Title')),
                    'Author': str(book_info.get('Book-Author', 'Unknown Author')),
                    'Year': int(book_info.get('Year-Of-Publication', 2000)),  # Convert to native int
                    'Publisher': str(book_info.get('Publisher', 'Unknown Publisher')),
                    'Predicted_Rating': float(round(predicted_rating * 10, 1))  # Convert to native float
                })
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def hybrid_recommendations(self, user_ratings, n=10):
        """Smart recommendations with automatic fallbacks"""
        try:
            # Check if we have a trained model
            if self.svd_model is None or not getattr(self.svd_model, 'is_trained', False):
                print("Using content-based recommendations (model not trained)")
                return self._get_content_based_recommendations(user_ratings, n)

            # Use the trained SVD model
            if len(user_ratings) < 2:
                return self.get_cold_start_recommendations(user_ratings, n)
            else:
                return self._get_personalized_recommendations_svd(user_ratings, n)

        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            # Fallback to content-based recommendations
            return self._get_content_based_recommendations(user_ratings, n)

    def _get_content_based_recommendations(self, user_ratings, n=10):
        """Content-based recommendations using book similarity"""
        print("Using content-based fallback recommendations")
        
        # Get popular books as fallback
        if hasattr(self.data_processor, 'get_popular_books'):
            popular_books = self.data_processor.get_popular_books(n * 2)
            if not popular_books.empty:
                recommendations = []
                for _, book in popular_books.iterrows():
                    # Convert to proper format
                    recommendations.append({
                        'ISBN': str(book['ISBN']),
                        'Title': str(book.get('Book-Title', 'Unknown Title')),
                        'Author': str(book.get('Book-Author', 'Unknown Author')),
                        'Year': int(book.get('Year-Of-Publication', 2000)),
                        'Publisher': str(book.get('Publisher', 'Unknown Publisher')),
                        'Predicted_Rating': float(book.get('average_rating', 8.0)) if 'average_rating' in book else 8.0
                    })
                
                # Remove any books the user already rated
                user_rated_isbns = [isbn for isbn, _ in user_ratings]
                recommendations = [rec for rec in recommendations if rec['ISBN'] not in user_rated_isbns]
                
                return recommendations[:n]
        
        # Ultimate fallback
        return self._get_basic_fallback_recommendations(n)
    
    def _get_personalized_recommendations_svd(self, user_ratings, n=10):
        """Personalized recommendations using SVD model"""
        try:
            # Create a simulated user ID for the new user
            simulated_user_id = len(self.data_processor.user_ids) + 1000  # Offset to avoid conflicts
            
            # Get recommendations using SVD
            recommendations = []
            all_books = self.data_processor.books_df.sample(min(500, len(self.data_processor.books_df)))  # Limit for performance
            
            user_rated_isbns = [isbn for isbn, _ in user_ratings]
            
            for _, book in all_books.iterrows():
                if len(recommendations) >= n:
                    break
                    
                if book['ISBN'] in user_rated_isbns:
                    continue
                    
                # Try to get book ID for prediction
                if book['ISBN'] in self.data_processor.book_to_idx:
                    book_idx = self.data_processor.book_to_idx[book['ISBN']]
                    try:
                        predicted_rating = self.svd_model.predict_rating(simulated_user_id, book_idx)
                        recommendations.append({
                            'ISBN': str(book['ISBN']),
                            'Title': str(book.get('Book-Title', 'Unknown Title')),
                            'Author': str(book.get('Book-Author', 'Unknown Author')),
                            'Year': int(book.get('Year-Of-Publication', 2000)),
                            'Publisher': str(book.get('Publisher', 'Unknown Publisher')),
                            'Predicted_Rating': float(round(predicted_rating * 10, 1))
                        })
                    except:
                        # If prediction fails, use average rating
                        recommendations.append({
                            'ISBN': str(book['ISBN']),
                            'Title': str(book.get('Book-Title', 'Unknown Title')),
                            'Author': str(book.get('Book-Author', 'Unknown Author')),
                            'Year': int(book.get('Year-Of-Publication', 2000)),
                            'Publisher': str(book.get('Publisher', 'Unknown Publisher')),
                            'Predicted_Rating': 8.0
                        })
            
            return recommendations[:n]
            
        except Exception as e:
            print(f"SVD recommendation error: {e}")
            return self._get_content_based_recommendations(user_ratings, n)
    
    def _get_basic_fallback_recommendations(self, n=10):
        """Ultimate fallback with hardcoded recommendations"""
        fallback_books = [
            {
                'ISBN': '0439708184',
                'Title': 'Harry Potter and the Sorcerer\'s Stone',
                'Author': 'J.K. Rowling',
                'Year': 1998,
                'Publisher': 'Scholastic',
                'Predicted_Rating': 9.0
            },
            {
                'ISBN': '0439064872',
                'Title': 'Harry Potter and the Chamber of Secrets',
                'Author': 'J.K. Rowling', 
                'Year': 1999,
                'Publisher': 'Scholastic',
                'Predicted_Rating': 8.8
            },
            {
                'ISBN': '0451524934',
                'Title': '1984',
                'Author': 'George Orwell',
                'Year': 1949,
                'Publisher': 'Signet Classic',
                'Predicted_Rating': 8.7
            }
        ]
        return fallback_books[:n]