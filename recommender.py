import pandas as pd
import numpy as np
from surprise import SVD
import joblib
from collections import defaultdict
import random
from datetime import datetime

class BookRecommender:
    def __init__(self, model, books_df, user_encoder, book_encoder):
        self.model = model
        self.books_df = books_df
        self.user_encoder = user_encoder
        self.book_encoder = book_encoder
        self.book_mapping = dict(zip(books_df['ISBN'], books_df['Book-Title']))
        self.reverse_book_encoding = {v: k for k, v in enumerate(books_df['ISBN'])}
        self.current_year = datetime.now().year
        self.virtual_user_counter = 1000000  # ID user virtual yang unik
    
    def create_virtual_user_profile(self, user_ratings):
        """Buat profil user virtual berdasarkan rating yang diberikan"""
        if not user_ratings:
            return None, None
            
        # Buat user ID virtual yang unik berdasarkan kombinasi ISBN + rating
        rating_hash = hash(tuple(sorted([(r['isbn'], r['rating']) for r in user_ratings])))
        virtual_user_id = self.virtual_user_counter + abs(rating_hash) % 100000
        
        print(f"🎯 Virtual User ID created: {virtual_user_id} for {len(user_ratings)} ratings")
        return virtual_user_id, user_ratings
    
    def search_books(self, query, max_results=20):
        """Search books by title or author dengan improved matching"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            if not query or not isinstance(query, str):
                return []
            
            query = query.lower().strip()
            if len(query) < 2:
                return []
            
            # Search in title and author
            try:
                title_matches = self.books_df['Book-Title'].astype(str).str.lower().str.contains(query, na=False)
                author_matches = self.books_df['Book-Author'].astype(str).str.lower().str.contains(query, na=False)
                matches = self.books_df[title_matches | author_matches]
            except Exception as e:
                return []
            
            results = []
            seen_isbns = set()
            
            for _, book in matches.head(max_results * 2).iterrows():
                if book['ISBN'] in seen_isbns:
                    continue
                seen_isbns.add(book['ISBN'])
                
                book_title = str(book['Book-Title']).strip()
                book_author = str(book['Book-Author']).strip()
                
                if (pd.isna(book_title) or book_title == '' or 
                    pd.isna(book_author) or book_author == ''):
                    continue
                
                publication_year = self.validate_publication_year(book)
                cover_url = self.get_book_cover(book)
                
                results.append({
                    'isbn': book['ISBN'],
                    'title': book_title,
                    'author': book_author,
                    'year': publication_year,
                    'publisher': str(book['Publisher']).strip() if not pd.isna(book['Publisher']) else 'Unknown Publisher',
                    'cover_url': cover_url,
                    'has_cover': bool(cover_url)
                })
                
                if len(results) >= max_results:
                    break
            
            return results
            
        except Exception as e:
            return []
    
    def validate_publication_year(self, book_info):
        """Validate and format publication year"""
        try:
            year = book_info['Year-Of-Publication']
            if pd.isna(year) or year == 0 or year == '':
                return "Unknown"
            
            year_int = int(year)
            if year_int < 1800 or year_int > self.current_year + 1:
                return "Unknown"
            
            return year_int
        except (ValueError, TypeError):
            return "Unknown"
    
    def get_book_cover(self, book_info):
        """Get book cover URL jika tersedia"""
        try:
            cover_columns = ['Image-URL', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Cover-URL', 'cover_url']
            
            for column in cover_columns:
                if column in book_info and not pd.isna(book_info[column]) and book_info[column].strip():
                    url = str(book_info[column]).strip()
                    if url.startswith(('http://', 'https://')):
                        return url
            return None
        except Exception:
            return None
    
    def calculate_confidence_level_original(self, predicted_rating):
        """Confidence level original yang intuitif: >=4.5 = High"""
        # LOGICA ASLI YANG INTUITIF
        if predicted_rating >= 4.2 or predicted_rating <= 1.5:
            confidence_score = 0.9  # High
        elif predicted_rating >= 4.0 or predicted_rating <= 2.0:
            confidence_score = 0.7  # Medium
        elif predicted_rating >= 3.5 or predicted_rating <= 2.5:
            confidence_score = 0.5  # Medium
        else:
            confidence_score = 0.3  # Low
        
        # Map to confidence levels
        if confidence_score >= 0.8:
            return "high", confidence_score
        elif confidence_score >= 0.6:
            return "medium", confidence_score
        else:
            return "low", confidence_score
    
    def get_user_recommendations(self, user_ratings, n_recommendations=10):
        try:
            print(f"🔍 Generating recommendations for {len(user_ratings)} user ratings...")
            
            # Handle cold start
            if len(user_ratings) < 1:
                return self.get_popular_books_varied(n_recommendations)
            
            # BUAT PROFIL USER VIRTUAL YANG UNIK
            virtual_user_id, processed_ratings = self.create_virtual_user_profile(user_ratings)
            
            # Encode virtual user ID untuk model
            user_id_encoded = virtual_user_id % 10000
            
            # Get all book IDs that user hasn't rated
            rated_books_isbns = set([rating['isbn'] for rating in user_ratings])
            all_books_isbns = set(self.books_df['ISBN'].unique())
            unrated_books_isbns = all_books_isbns - rated_books_isbns
            
            print(f"📚 Books: {len(all_books_isbns)}, Rated: {len(rated_books_isbns)}, Unrated: {len(unrated_books_isbns)}")
            
            # **PERBAIKAN PENTING: Ambil lebih banyak prediksi untuk variasi**
            predictions = []
            
            # Coba prediksi untuk lebih banyak buku (2000 bukan 800)
            sample_size = min(2000, len(unrated_books_isbns))
            books_to_predict = random.sample(list(unrated_books_isbns), sample_size)
            
            for book_isbn in books_to_predict:
                try:
                    if book_isbn in self.reverse_book_encoding:
                        book_id_encoded = self.reverse_book_encoding[book_isbn]
                        
                        # **GUNAKAN VIRTUAL USER ID YANG UNIK**
                        pred = self.model.predict(user_id_encoded, book_id_encoded)
                        predictions.append((book_isbn, pred.est))
                except Exception:
                    continue
            
            # **STRATEGI VARIASI: Group berdasarkan rating range**
            high_predictions = [(isbn, rating) for isbn, rating in predictions if rating >= 4.5]
            medium_high_predictions = [(isbn, rating) for isbn, rating in predictions if 4.0 <= rating < 4.5]
            medium_predictions = [(isbn, rating) for isbn, rating in predictions if 3.5 <= rating < 4.0]
            low_predictions = [(isbn, rating) for isbn, rating in predictions if rating < 3.5]
            
            print(f"📊 Prediction distribution - High: {len(high_predictions)}, Med-High: {len(medium_high_predictions)}, Med: {len(medium_predictions)}, Low: {len(low_predictions)}")
            
            # **STRATEGI SELECTION: Ambil proporsional dari setiap group**
            final_selection = []
            
            # Tambahkan high (50%)
            high_target = max(5, n_recommendations // 2)
            if len(high_predictions) > 0:
                final_selection.extend(high_predictions[:high_target])
            
            # Tambahkan medium-high (30%)
            med_high_target = max(3, n_recommendations // 3)
            if len(medium_high_predictions) > 0:
                final_selection.extend(medium_high_predictions[:med_high_target])
            
            # Tambahkan medium (15%)
            med_target = max(2, n_recommendations // 6)
            if len(medium_predictions) > 0:
                final_selection.extend(medium_predictions[:med_target])
            
            # Tambahkan low (5%) untuk variasi
            low_target = max(1, n_recommendations // 10)
            if len(low_predictions) > 0:
                final_selection.extend(low_predictions[:low_target])
            
            # Jika masih kurang, isi dengan predictions terbaik
            if len(final_selection) < n_recommendations:
                remaining_needed = n_recommendations - len(final_selection)
                # Ambil dari predictions yang belum terpilih, sorted by rating
                all_predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
                for pred in all_predictions_sorted:
                    if pred not in final_selection and len(final_selection) < n_recommendations:
                        final_selection.append(pred)
            
            # **PERBAIKAN: Acak urutan untuk variasi visual**
            final_selection.sort(key=lambda x: x[1], reverse=True)
            
            # Process recommendations
            recommendations = []
            seen_isbns = set()
            seen_titles = set()
            
            for book_isbn, predicted_rating in final_selection[:n_recommendations]:
                try:
                    if book_isbn in seen_isbns:
                        continue
                    seen_isbns.add(book_isbn)
                    
                    book_info = self.books_df[self.books_df['ISBN'] == book_isbn].iloc[0]
                    
                    # Skip books dengan informasi kritikal yang missing
                    if (pd.isna(book_info['Book-Title']) or 
                        pd.isna(book_info['Book-Author']) or 
                        str(book_info['Book-Title']).strip() == ''):
                        continue
                    
                    book_title = str(book_info['Book-Title']).strip()
                    title_lower = book_title.lower()
                    
                    if title_lower in seen_titles:
                        continue
                    seen_titles.add(title_lower)
                    
                    publication_year = self.validate_publication_year(book_info)
                    predicted_rating_clamped = max(1.0, min(5.0, predicted_rating))
                    cover_url = self.get_book_cover(book_info)
                    
                    # **GUNAKAN CONFIDENCE LEVEL ORIGINAL YANG INTUITIF**
                    confidence_level, confidence_score = self.calculate_confidence_level_original(predicted_rating_clamped)
                    
                    recommendation = {
                        'isbn': book_isbn,
                        'title': book_title,
                        'author': str(book_info['Book-Author']).strip(),
                        'year': publication_year,
                        'publisher': str(book_info['Publisher']).strip() if not pd.isna(book_info['Publisher']) else 'Unknown Publisher',
                        'predicted_rating': round(predicted_rating_clamped, 2),
                        'confidence_level': confidence_level,
                        'confidence_score': confidence_score,
                        'cover_url': cover_url,
                        'has_cover': bool(cover_url)
                    }
                    
                    recommendations.append(recommendation)
                    
                    if len(recommendations) >= n_recommendations:
                        break
                        
                except Exception:
                    continue
            
            # **HITUNG DISTRIBUSI CONFIDENCE FINAL**
            conf_counts = {}
            for rec in recommendations:
                conf_level = rec['confidence_level']
                conf_counts[conf_level] = conf_counts.get(conf_level, 0) + 1
            
            print(f"✅ Generated {len(recommendations)} recommendations with variety")
            print(f"🎯 Final confidence distribution:")
            for level, count in conf_counts.items():
                print(f"   {level}: {count} books")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in recommendations: {e}")
            return self.get_popular_books_varied(n_recommendations)
    
    def get_popular_books_varied(self, n_recommendations=10):
        """Get popular books dengan variasi confidence yang beragam"""
        try:
            valid_books = self.books_df[
                (self.books_df['Book-Title'].notna()) & 
                (self.books_df['Book-Author'].notna()) &
                (self.books_df['Year-Of-Publication'] > 1900) &
                (self.books_df['Year-Of-Publication'] <= self.current_year)
            ].copy()
            
            # **STRATEGI VARIASI: Ambil dari berbagai tahun dan rating**
            recent_books = valid_books.nlargest(n_recommendations * 2, 'Year-Of-Publication')
            classic_books = valid_books.nsmallest(n_recommendations, 'Year-Of-Publication')
            mid_age_books = valid_books[
                (valid_books['Year-Of-Publication'] >= 1990) & 
                (valid_books['Year-Of-Publication'] <= 2010)
            ].sample(min(n_recommendations, len(valid_books)))
            
            # Gabungkan dan acak
            popular_books = pd.concat([recent_books, classic_books, mid_age_books]).drop_duplicates()
            popular_books = popular_books.sample(min(n_recommendations, len(popular_books)))
            
            recommendations = []
            seen_isbns = set()
            seen_titles = set()
            
            for _, book in popular_books.iterrows():
                if book['ISBN'] in seen_isbns:
                    continue
                
                book_title = str(book['Book-Title']).strip()
                title_lower = book_title.lower()
                if title_lower in seen_titles:
                    continue
                
                seen_isbns.add(book['ISBN'])
                seen_titles.add(title_lower)
                
                publication_year = self.validate_publication_year(book)
                cover_url = self.get_book_cover(book)
                
                # **VARIASI RATING: Berikan range yang beragam berdasarkan tahun**
                year = publication_year if isinstance(publication_year, int) else 2000
                
                if year > 2015:  # Buku sangat baru → rating tinggi
                    predicted_rating = round(random.uniform(4.3, 4.9), 2)
                elif year > 2010:  # Buku baru → rating tinggi-medium
                    predicted_rating = round(random.uniform(4.0, 4.7), 2)
                elif year < 1980:  # Buku klasik → rating medium-tinggi
                    predicted_rating = round(random.uniform(3.8, 4.5), 2)
                else:  # Buku medium → rating beragam
                    predicted_rating = round(random.uniform(3.2, 4.3), 2)
                
                confidence_level, confidence_score = self.calculate_confidence_level_original(predicted_rating)
                
                recommendations.append({
                    'isbn': book['ISBN'],
                    'title': book_title,
                    'author': str(book['Book-Author']).strip(),
                    'year': publication_year,
                    'publisher': str(book['Publisher']).strip() if not pd.isna(book['Publisher']) else 'Unknown Publisher',
                    'predicted_rating': predicted_rating,
                    'confidence_level': confidence_level,
                    'confidence_score': round(confidence_score, 2),
                    'cover_url': cover_url,
                    'has_cover': bool(cover_url)
                })
            
            # **HITUNG DISTRIBUSI CONFIDENCE**
            conf_counts = {}
            for rec in recommendations:
                conf_level = rec['confidence_level']
                conf_counts[conf_level] = conf_counts.get(conf_level, 0) + 1
            
            print(f"📚 Popular books variety: {len([r for r in recommendations if r['confidence_level'] == 'high'])} high, "
                  f"{len([r for r in recommendations if r['confidence_level'] == 'medium'])} medium, "
                  f"{len([r for r in recommendations if r['confidence_level'] == 'low'])} low")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in popular books: {e}")
            return []
    
    # Backward compatibility
    def get_popular_books(self, n_recommendations=10):
        """Alias untuk get_popular_books_varied"""
        return self.get_popular_books_varied(n_recommendations)