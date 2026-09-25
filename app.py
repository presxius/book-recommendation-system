from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import numpy as np
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from recommender import BookRecommender
from utils import format_metrics, NumpyEncoder
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'book_recommendation_secret_key'
app.json_encoder = NumpyEncoder

# Global variables
data_processor = None
model_trainer = None
recommender = None
books_df = None

def initialize_system():
    """Initialize the recommendation system"""
    global data_processor, model_trainer, recommender, books_df
    
    try:
        # Initialize data processor
        data_processor = DataPreprocessor()
        
        # Load and preprocess data
        data_loaded = data_processor.load_data(
            'data/Books.csv',
            'data/Users.csv', 
            'data/Ratings.csv'
        )
        
        if not data_loaded:
            print("❌ Failed to load data")
            return False
        
        # Preprocess data
        books_df = data_processor.preprocess_books()
        data_processor.preprocess_users()
        ratings_df = data_processor.preprocess_ratings()
        
        # Initialize model trainer
        model_trainer = ModelTrainer(data_processor)
        
        # Check if model exists, otherwise train
        model_path = 'models/svd_model.pkl'
        if os.path.exists(model_path):
            print("Loading existing model...")
            model = joblib.load(model_path)
            model_trainer.model = model
            
            # Load metrics
            metrics = load_metrics()
            if metrics:
                model_trainer.metrics = metrics
                print("✅ Metrics file loaded successfully")
        else:
            print("Training new model...")
            model = model_trainer.train_model(ratings_df)
            # Simpan metrics setelah training awal
            if model_trainer.metrics:
                save_metrics(model_trainer.metrics)
        
        # Initialize recommender
        preprocessed_data = data_processor.get_preprocessed_data()
        recommender = BookRecommender(
            model_trainer.model,
            books_df,
            preprocessed_data['user_encoder'],
            preprocessed_data['book_encoder']
        )
        
        # Test search functionality during initialization
        print("🧪 Testing search functionality...")
        test_results = recommender.search_books("test", 2)
        print(f"✅ Search test returned {len(test_results)} results")
        
        print("✅ System initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_metrics(metrics):
    """Save metrics to JSON file"""
    try:
        os.makedirs('models', exist_ok=True)
        
        metrics_serializable = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                metrics_serializable[key] = value.item()
            elif isinstance(value, (np.floating, np.integer)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        metrics_serializable['last_updated'] = datetime.now().isoformat()
        
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print("✅ Metrics saved successfully")
        return True
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
        return False

def load_metrics():
    """Load metrics from JSON file"""
    try:
        metrics_path = 'models/metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

# app.py - Perbaiki endpoint search_books dengan debugging
@app.route('/search', methods=['POST'])
def search_books():
    """Search for books"""
    try:
        query = request.json.get('query', '')
        print(f"🔍 Search query received: '{query}'")
        
        if not query or not recommender:
            print("❌ No query or recommender not available")
            return jsonify({'results': []})
        
        results = recommender.search_books(query)
        print(f"✅ Search found {len(results)} results for query: '{query}'")
        
        # Debug: print first few results if any
        if results:
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. {result['title']} by {result['author']}")
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"❌ Error in search_books: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'results': []})

# app.py - Perbaiki route /recommend untuk handle recommendations dengan lebih baik

@app.route('/recommend', methods=['GET', 'POST'])
def get_recommendations():
    """Get book recommendations - VERSI DIPERBAIKI"""
    try:
        if request.method == 'POST':
            # Get user ratings from form
            user_ratings = []
            rated_books = {}
            
            for key, value in request.form.items():
                if key.startswith('rating_'):
                    isbn = key.replace('rating_', '')
                    rating = int(value)
                    book_title = request.form.get(f'title_{isbn}', '')
                    
                    if rating > 0:
                        rated_books[isbn] = {
                            'isbn': isbn,
                            'rating': rating,
                            'title': book_title
                        }
            
            # Convert dictionary back to list
            user_ratings = list(rated_books.values())
            
            # Lengkapi judul buku yang missing
            user_ratings = complete_book_titles(user_ratings)
            
            # Store ratings in session
            session['user_ratings'] = user_ratings
            
            print(f"📝 User ratings saved: {len(user_ratings)} books")
            for rating in user_ratings:
                print(f"  - {rating['title']}: {rating['rating']}/5")
            
        else:
            # GET request - get ratings from session
            user_ratings = session.get('user_ratings', [])
            user_ratings = complete_book_titles(user_ratings)
        
        # Get recommendations
        if not recommender:
            return render_template('recommend.html', 
                                recommendations=[], 
                                user_ratings=user_ratings,
                                error="System not initialized")
        
        print(f"🎯 Getting recommendations based on {len(user_ratings)} user ratings...")
        
        # PANGGIL REKOMMENDER YANG SUDAH DIPERBAIKI
        recommendations = recommender.get_user_recommendations(user_ratings, 12)
        
        # Sort recommendations by predicted rating (highest first)
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        print(f"✅ Successfully generated {len(recommendations)} recommendations")
        print(f"🔍 Recommendation differences:")
        for i, rec in enumerate(recommendations[:5]):
            print(f"  {i+1}. {rec['title']} - {rec['predicted_rating']}/5 (confidence: {rec['confidence_level']})")
        
        return render_template('recommend.html', 
                             recommendations=recommendations,
                             user_ratings=user_ratings)
    
    except Exception as e:
        print(f"❌ Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return render_template('recommend.html', 
                             recommendations=[], 
                             user_ratings=session.get('user_ratings', []),
                             error=f"Error generating recommendations: {str(e)}")

def complete_book_titles(user_ratings):
    """Lengkapi judul buku yang missing dari books_df berdasarkan ISBN"""
    if not user_ratings or not hasattr(data_processor, 'books_df'):
        return user_ratings
    
    try:
        completed_ratings = []
        for rating in user_ratings:
            isbn = rating['isbn']
            current_title = rating.get('title', '')
            
            if not current_title or current_title == 'Unknown Book':
                book_match = data_processor.books_df[data_processor.books_df['ISBN'] == isbn]
                if not book_match.empty:
                    new_title = book_match.iloc[0]['Book-Title']
                    rating['title'] = new_title
                else:
                    rating['title'] = f"Book (ISBN: {isbn[:10]}...)"
            
            completed_ratings.append(rating)
        
        return completed_ratings
        
    except Exception as e:
        return user_ratings

@app.route('/reset_session')
def reset_session():
    """Reset user session - clear all ratings"""
    try:
        # Clear user ratings from session
        if 'user_ratings' in session:
            old_count = len(session['user_ratings'])
            session.pop('user_ratings')
            print(f"Session reset: cleared {old_count} ratings")
        else:
            print("Session reset: no ratings to clear")
        
        # Optional: clear other session data if needed
        # session.clear()
        
        # Redirect back to recommendations page with success message
        return redirect(url_for('get_recommendations', reset='success'))
    
    except Exception as e:
        print(f"Error resetting session: {e}")
        return redirect(url_for('get_recommendations', reset='error'))

# app.py - Perbaiki fungsi train() dan pastikan data_stats selalu terkirim
@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train or retrain the model - SEKALIGUS MENAMPILKAN METRICS"""
    try:
        # Load metrics terlebih dahulu
        metrics = load_metrics()
        formatted_metrics = format_metrics(metrics) if metrics else {}
        
        # PASTIKAN data_stats SELALU ADA - panggil fungsi get_data_statistics()
        data_stats = get_data_statistics()
        
        if request.method == 'POST':
            force_retrain = request.form.get('force_retrain') == 'true'
            
            if not data_processor:
                return render_template('train.html', 
                                     metrics=formatted_metrics,
                                     data_stats=data_stats,
                                     error="Data processor not initialized")
            
            ratings_df = data_processor.preprocess_ratings()
            model_trainer.train_model(ratings_df, force_retrain=force_retrain)
            
            # Simpan metrics setelah training
            if model_trainer.metrics:
                save_metrics(model_trainer.metrics)
                # Update metrics yang akan ditampilkan
                metrics = model_trainer.metrics
                formatted_metrics = format_metrics(metrics)
            
            # Update data statistik setelah training
            data_stats = get_data_statistics()
            
            # Update recommender
            preprocessed_data = data_processor.get_preprocessed_data()
            global recommender
            recommender = BookRecommender(
                model_trainer.model,
                books_df,
                preprocessed_data['user_encoder'],
                preprocessed_data['book_encoder']
            )
            
            return render_template('train.html', 
                                 metrics=formatted_metrics,
                                 data_stats=data_stats,
                                 message="Model trained successfully!")
        
        # GET request - show training page dengan metrics terbaru
        return render_template('train.html', 
                             metrics=formatted_metrics,
                             data_stats=data_stats,
                             message="Model metrics loaded" if metrics else "No metrics available")
    
    except Exception as e:
        print(f"Error in train route: {e}")
        # PASTIKAN data_stats juga dikirim saat error
        data_stats = get_data_statistics()
        return render_template('train.html', 
                             metrics={}, 
                             data_stats=data_stats,
                             error=f"Training failed: {str(e)}")

def get_data_statistics():
    """Get data statistics for display - PASTIKAN FUNGSI INI SELALU RETURN DICT"""
    try:
        if data_processor and data_processor.books_df is not None:
            stats = {
                'books_count': len(data_processor.books_df),
                'users_count': len(data_processor.users_df) if data_processor.users_df is not None else 0,
                'ratings_count': len(data_processor.ratings_df) if data_processor.ratings_df is not None else 0
            }
            
            if data_processor.ratings_df is not None and not data_processor.ratings_df.empty:
                stats.update({
                    'min_rating': float(data_processor.ratings_df['Book-Rating'].min()),
                    'max_rating': float(data_processor.ratings_df['Book-Rating'].max()),
                    'avg_rating': float(data_processor.ratings_df['Book-Rating'].mean().round(2))
                })
            else:
                stats.update({
                    'min_rating': 1.0,
                    'max_rating': 5.0,
                    'avg_rating': 3.0
                })
            
            print(f"Data statistics: {stats}")  # Debug print
            return stats
        else:
            # Return default values jika data_processor belum diinisialisasi
            default_stats = {
                'books_count': 0,
                'users_count': 0,
                'ratings_count': 0,
                'min_rating': 1.0,
                'max_rating': 5.0,
                'avg_rating': 3.0
            }
            print(f"Using default statistics: {default_stats}")  # Debug print
            return default_stats
            
    except Exception as e:
        print(f"Error getting data statistics: {e}")
        # Return default values jika ada error
        return {
            'books_count': 0,
            'users_count': 0,
            'ratings_count': 0,
            'min_rating': 1.0,
            'max_rating': 5.0,
            'avg_rating': 3.0
        }
    
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json
        user_ratings = data.get('ratings', [])
        
        if not recommender:
            return jsonify({'error': 'Recommender not initialized'})
        
        recommendations = recommender.get_user_recommendations(user_ratings, 10)
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Initializing Book Recommendation System...")
    success = initialize_system()
    
    if success:
        print("✅ System initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)

# app.py - Tambahkan fungsi untuk mendapatkan data statistik
@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train or retrain the model - SEKALIGUS MENAMPILKAN METRICS"""
    try:
        # Load metrics terlebih dahulu (untuk GET request dan POST request setelah training)
        metrics = load_metrics()
        formatted_metrics = format_metrics(metrics) if metrics else {}
        
        # Dapatkan data statistik untuk ditampilkan
        data_stats = get_data_statistics()
        
        if request.method == 'POST':
            force_retrain = request.form.get('force_retrain') == 'true'
            
            if not data_processor:
                return jsonify({'error': 'Data processor not initialized'})
            
            ratings_df = data_processor.preprocess_ratings()
            model_trainer.train_model(ratings_df, force_retrain=force_retrain)
            
            # Simpan metrics setelah training
            if model_trainer.metrics:
                save_metrics(model_trainer.metrics)
                # Update metrics yang akan ditampilkan
                metrics = model_trainer.metrics
                formatted_metrics = format_metrics(metrics)
            
            # Update data statistik setelah training
            data_stats = get_data_statistics()
            
            # Update recommender
            preprocessed_data = data_processor.get_preprocessed_data()
            global recommender
            recommender = BookRecommender(
                model_trainer.model,
                books_df,
                preprocessed_data['user_encoder'],
                preprocessed_data['book_encoder']
            )
            
            return render_template('train.html', 
                                 metrics=formatted_metrics,
                                 data_stats=data_stats,
                                 message="Model trained successfully!")
        
        # GET request - show training page dengan metrics terbaru
        return render_template('train.html', 
                             metrics=formatted_metrics,
                             data_stats=data_stats,
                             message="Model metrics loaded" if metrics else "No metrics available")
    
    except Exception as e:
        return render_template('train.html', 
                             metrics={}, 
                             data_stats={},
                             error=f"Training failed: {str(e)}")

def get_data_statistics():
    """Get data statistics for display"""
    try:
        if data_processor and data_processor.books_df is not None:
            stats = {
                'books_count': len(data_processor.books_df),
                'users_count': len(data_processor.users_df) if data_processor.users_df is not None else 0,
                'ratings_count': len(data_processor.ratings_df) if data_processor.ratings_df is not None else 0
            }
            
            if data_processor.ratings_df is not None and not data_processor.ratings_df.empty:
                stats.update({
                    'min_rating': float(data_processor.ratings_df['Book-Rating'].min()),
                    'max_rating': float(data_processor.ratings_df['Book-Rating'].max()),
                    'avg_rating': float(data_processor.ratings_df['Book-Rating'].mean().round(2))
                })
            else:
                stats.update({
                    'min_rating': 1.0,
                    'max_rating': 5.0,
                    'avg_rating': 3.0
                })
            
            return stats
        else:
            return {
                'books_count': 0,
                'users_count': 0,
                'ratings_count': 0,
                'min_rating': 1.0,
                'max_rating': 5.0,
                'avg_rating': 3.0
            }
    except Exception as e:
        print(f"Error getting data statistics: {e}")
        return {
            'books_count': 0,
            'users_count': 0,
            'ratings_count': 0,
            'min_rating': 1.0,
            'max_rating': 5.0,
            'avg_rating': 3.0
        }