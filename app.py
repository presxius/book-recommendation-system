from flask import Flask, render_template, request, jsonify, session
import os
import json
import numpy as np
import pandas as pd
from models.data_processor import DataProcessor
from models.svd_model import SVDAutoRec
from models.recommendation_engine import RecommendationEngine

# Custom JSON encoder class
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

app = Flask(__name__)
app.secret_key = 'book-recommendation-secret-key-2024'
app.json_encoder = NumpyJSONEncoder

# Initialize components
data_processor = DataProcessor()
svd_model = SVDAutoRec()
recommendation_engine = None

def initialize_system():
    """Initialize the recommendation system"""
    global recommendation_engine
    
    try:
        print("Initializing recommendation system...")
        
        # Load data
        data_loaded = data_processor.load_data(
            'data/books.csv',
            'data/ratings.csv', 
            'data/users.csv'
        )
        
        if not data_loaded:
            print("Failed to load data")
            return False
        
        print("Data loaded successfully, continuing with preprocessing...")
        
        # Preprocess data with error handling
        try:
            ratings_df, books_df = data_processor.preprocess_data()
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            print("Attempting to continue with basic processing...")
            # Basic processing without advanced features
            data_processor.handle_missing_values()
            data_processor.normalize_data()
            if hasattr(data_processor, '_filter_sparse_data'):
                data_processor._filter_sparse_data()
            if hasattr(data_processor, '_encode_ids'):
                data_processor._encode_ids()
            ratings_df, books_df = data_processor.ratings_df, data_processor.books_df
        
        if ratings_df is None or ratings_df.empty:
            print("No ratings data available after preprocessing")
            return False
        
        print(f"Data ready: {len(ratings_df)} ratings, {len(books_df)} books")
        
        # Get data in Surprise format
        surprise_data = data_processor.get_surprise_data()
        
        if surprise_data is None:
            print("Failed to prepare Surprise data")
            return False
        
        # Train SVD model
        print("Starting model training...")
        rmse, mae = svd_model.train(surprise_data)
        print(f"Model training completed! RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(data_processor, svd_model)
        
        print("System initialized successfully!")
        print(f"Available users: {len(data_processor.user_ids)}")
        print(f"Available books: {len(data_processor.book_ids)}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Home page with book selection"""
    try:
        # Get popular books for initial selection
        popular_books = data_processor.get_popular_books(100)
        
        if popular_books.empty:
            books_list = []
            print("No popular books available")
        else:
            books_list = popular_books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']].to_dict('records')
            print(f"Loaded {len(books_list)} popular books for selection")
        
        return render_template('index.html', books=books_list)
    
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', books=[])

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations based on user ratings"""
    try:
        user_ratings = request.json.get('ratings', [])
        
        if not user_ratings:
            return jsonify({'error': 'No ratings provided'}), 400
        
        print(f"Received {len(user_ratings)} ratings from user")
        
        if recommendation_engine is None:
            return jsonify({'error': 'Recommendation engine not initialized'}), 500
        
        # Get recommendations
        recommendations = recommendation_engine.hybrid_recommendations(
            user_ratings, 
            n=10
        )
        
        print(f"Generated {len(recommendations)} recommendations")
        
        # Convert all numpy/pandas types to native Python types for JSON serialization
        serializable_recommendations = []
        for rec in recommendations:
            serializable_rec = {
                'ISBN': str(rec['ISBN']),
                'Title': str(rec['Title']),
                'Author': str(rec['Author']),
                'Year': int(rec['Year']) if pd.notna(rec['Year']) else 2000,
                'Publisher': str(rec['Publisher']),
                'Predicted_Rating': float(rec['Predicted_Rating'])
            }
            serializable_recommendations.append(serializable_rec)
        
        # Store user ratings in session
        session['user_ratings'] = user_ratings
        
        return jsonify({
            'success': True,
            'recommendations': serializable_recommendations
        })
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations')
def show_recommendations():
    """Display recommendations page"""
    return render_template('recommendations.html')

@app.route('/search_books')
def search_books():
    """Search books by title or author"""
    try:
        query = request.args.get('q', '').lower().strip()
        
        if not query or len(query) < 2:
            return jsonify([])
        
        # Search in books dataframe
        matching_books = data_processor.search_books_deduplicated(query, 20)
        
        if matching_books.empty:
            return jsonify([])
        
        # Convert to serializable format
        results = []
        seen_titles = set()
        
        for _, book in matching_books.iterrows():
            # Create a unique identifier for the book (title + author)
            book_key = f"{book.get('Book-Title', '').lower()}_{book.get('Book-Author', '').lower()}"
            
            # Skip if we've already seen this book
            if book_key in seen_titles:
                continue
                
            seen_titles.add(book_key)
            
            results.append({
                'ISBN': str(book['ISBN']),
                'Book-Title': str(book.get('Book-Title', 'Unknown Title')),
                'Book-Author': str(book.get('Book-Author', 'Unknown Author')),
                'Year-Of-Publication': int(book.get('Year-Of-Publication', 2000)) if pd.notna(book.get('Year-Of-Publication')) else 2000,
                'Publisher': str(book.get('Publisher', 'Unknown Publisher'))
            })
            
            if len(results) >= 15:
                break
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error searching books: {e}")
        return jsonify([])

@app.route('/book_details/<isbn>')
def book_details(isbn):
    """Get detailed information about a book"""
    try:
        book_info = data_processor.books_df[data_processor.books_df['ISBN'] == isbn]
        
        if book_info.empty:
            return jsonify({'error': 'Book not found'}), 404
        
        book_data = book_info.iloc[0].to_dict()
        return jsonify(book_data)
    
    except Exception as e:
        print(f"Error getting book details: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    system_ready = recommendation_engine is not None and svd_model.is_trained
    return jsonify({
        'status': 'ready' if system_ready else 'initializing',
        'users_loaded': len(data_processor.user_ids) if hasattr(data_processor, 'user_ids') else 0,
        'books_loaded': len(data_processor.book_ids) if hasattr(data_processor, 'book_ids') else 0,
        'model_trained': svd_model.is_trained
    })

if __name__ == '__main__':
    # Initialize the system when starting the app
    print("Starting Book Recommendation System...")
    if initialize_system():
        print("Application starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize the recommendation system!")
        print("Starting application in limited mode...")
        app.run(debug=True, host='0.0.0.0', port=5000)