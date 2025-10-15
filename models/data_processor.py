import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import re
from difflib import SequenceMatcher

class DataProcessor:
    def __init__(self):
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.books_deduplicated = None
        
    def load_data(self, books_path, ratings_path, users_path):
        """Load and preprocess the dataset with proper encoding handling"""
        try:
            # Try different encodings for books data
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    self.books_df = pd.read_csv(books_path, encoding=encoding, low_memory=False)
                    # Check if we have weird characters in the first few titles
                    sample_titles = self.books_df['Book-Title'].head(10).astype(str)
                    weird_chars = any('Ã' in title for title in sample_titles)
                    
                    if not weird_chars:
                        print(f"Success with encoding: {encoding}")
                        break
                    else:
                        print(f"Encoding {encoding} still has weird characters, trying next...")
                except UnicodeDecodeError:
                    print(f"Encoding {encoding} failed, trying next...")
                    continue
            
            # Load other files with standard encoding
            self.ratings_df = pd.read_csv(ratings_path, encoding='latin-1')
            self.users_df = pd.read_csv(users_path, encoding='latin-1')
            
            print("Original data shapes:")
            print(f"Books: {self.books_df.shape}")
            print(f"Ratings: {self.ratings_df.shape}")
            print(f"Users: {self.users_df.shape}")
            
            # Clean up encoding issues in book titles
            self._fix_encoding_issues()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _fix_encoding_issues(self):
        """Comprehensive encoding fix using multiple approaches"""
        print("Fixing encoding issues...")
        
        if self.books_df is None:
            return
            
        total_fixed = 0
        
        # Step 1: Apply pattern-based fixes
        total_fixed += self._apply_pattern_fixes()
        
        # Step 2: Apply manual fixes for specific known issues
        total_fixed += self._manual_specific_fixes()
        
        print(f"Total encoding fixes applied: {total_fixed}")
        
        # Show remaining issues
        self._show_remaining_encoding_issues()

    def _apply_pattern_fixes(self):
        """Apply regex pattern-based fixes"""
        fixed_count = 0
        
        encoding_fixes = [
            # Fix the specific patterns we're seeing (Ã? followed by something)
            (r'Ã\?Â¤', 'ä'),
            (r'Ã\?Â¶', 'ö'), 
            (r'Ã\?Â¼', 'ü'),
            (r'Ã\?Â', 'Ü'),
            (r'Ã\?Â', 'Ä'),
            (r'Ã\?Â', 'Ö'),
            (r'Ã\?Â', 'ß'),
            (r'Ã\?Â©', 'é'),
            (r'Ã\?Â¨', 'è'),
            (r'Ã\?Â§', 'ç'),
            (r'Ã\?Â±', 'ñ'),
            
            # Fix other common patterns
            (r'Ã¤', 'ä'),
            (r'Ã¶', 'ö'),
            (r'Ã¼', 'ü'),
            (r'Ã', 'Ä'),
            (r'Ã', 'Ö'), 
            (r'Ã', 'Ü'),
            (r'Ã', 'ß'),
            (r'Ã©', 'é'),
            (r'Ã¨', 'è'),
            (r'Ã§', 'ç'),
            (r'Ã±', 'ñ'),
        ]
        
        for col in ['Book-Title', 'Book-Author']:
            if col in self.books_df.columns:
                for bad, good in encoding_fixes:
                    try:
                        before_count = self.books_df[col].astype(str).str.contains(bad, na=False, regex=True).sum()
                        if before_count > 0:
                            self.books_df[col] = self.books_df[col].astype(str).str.replace(bad, good, regex=True)
                            fixed_count += before_count
                            print(f"Fixed {before_count} occurrences of '{bad}' -> '{good}' in {col}")
                    except Exception as e:
                        print(f"Error fixing pattern '{bad}': {e}")
                        continue
        
        return fixed_count

    def _manual_specific_fixes(self):
        """Apply manual fixes for specific known problematic titles"""
        manual_fixes = {
            # Your specific examples
            'Der illustrierte Mann. ErzÃ?Â¤hlungen.': 'Der illustrierte Mann. Erzählungen.',
            'Der KÃ?Â¶nig in Gelb.': 'Der König in Gelb.',
            'Die Mars- Chroniken. Roman in ErzÃ?Â¤hlungen.': 'Die Mars-Chroniken. Roman in Erzählungen.',
            
            # Common German words that appear frequently
            'ErzÃ?Â¤hlungen': 'Erzählungen',
            'KÃ?Â¶nig': 'König',
            'FÃ?Â¼r': 'Für',
            'fÃ?Â¼r': 'für',
            'Ã?Â¼ber': 'über',
        }
        
        fixed_count = 0
        for col in ['Book-Title', 'Book-Author']:
            if col in self.books_df.columns:
                for bad, good in manual_fixes.items():
                    mask = self.books_df[col].astype(str).str.contains(bad, na=False, regex=False)
                    if mask.any():
                        self.books_df.loc[mask, col] = self.books_df.loc[mask, col].str.replace(bad, good, regex=False)
                        fixed_count += mask.sum()
                        print(f"Manually fixed {mask.sum()} occurrences of '{bad}' -> '{good}' in {col}")
        
        return fixed_count

    def _show_remaining_encoding_issues(self):
        """Show remaining encoding issues for debugging - safe version"""
        if self.books_df is None:
            return
        
        # Safe patterns that won't cause regex issues
        weird_patterns = ['Ã', 'Â', '�', 'ð', '¡', '¢']
        
        remaining_issues = {}
        for pattern in weird_patterns:
            # Use simple string containment check instead of regex
            count = 0
            for title in self.books_df['Book-Title'].astype(str):
                if pattern in title:
                    count += 1
            if count > 0:
                remaining_issues[pattern] = count
        
        if remaining_issues:
            print(f"\nRemaining encoding issues by pattern:")
            for pattern, count in remaining_issues.items():
                print(f"  '{pattern}': {count} titles")
            
            # Show some examples without using problematic patterns in search
            print(f"\nSample of remaining problematic titles:")
            sample_count = 0
            for idx, row in self.books_df.iterrows():
                if sample_count >= 3:
                    break
                title = str(row['Book-Title'])
                if any(pattern in title for pattern in weird_patterns):
                    print(f"  {sample_count + 1}. {title}")
                    sample_count += 1

    def analyze_ratings_distribution(self):
        """Analyze and print rating distribution"""
        if self.ratings_df is None or 'Book-Rating' not in self.ratings_df.columns:
            return
        
        print("\n=== Rating Distribution Analysis ===")
        rating_counts = self.ratings_df['Book-Rating'].value_counts().sort_index()
        
        for rating, count in rating_counts.items():
            percentage = (count / len(self.ratings_df)) * 100
            print(f"Rating {rating}: {count:6d} ratings ({percentage:5.1f}%)")
        
        # Check for zeros specifically
        zero_count = (self.ratings_df['Book-Rating'] == 0).sum()
        if zero_count > 0:
            print(f"\nWARNING: Found {zero_count} zero ratings!")
            print("In book recommendation datasets, 0 often means 'no rating' rather than 'lowest rating'")
            print("Consider treating these as missing data rather than actual ratings.")

    def clean_book_title(self, title):
        """Clean and standardize book titles for better deduplication"""
        if pd.isna(title):
            return ""
        
        title = str(title).strip()
        
        # Remove common suffixes and edition indicators
        patterns_to_remove = [
            r'\([^)]*\)',  # Remove content in parentheses
            r'\[[^\]]*\]',  # Remove content in brackets
            r'\b(paperback|hardcover|ebook|kindle|edition)\b',
            r'\b(volume|vol\.)\b',
            r'\b(book \d+)\b',
            r'\b(\d+(st|nd|rd|th) edition)\b',
            r'\b(special|anniversary|collector\'s) edition\b',
        ]
        
        for pattern in patterns_to_remove:
            try:
                title = re.sub(pattern, '', title, flags=re.IGNORECASE)
            except re.error as e:
                continue
        
        # Remove special characters but keep spaces and basic punctuation
        title = re.sub(r'[^\w\s\-]', '', title)
        
        # Remove extra spaces and return
        title = re.sub(r'\s+', ' ', title).strip()
        return title.lower()

    def deduplicate_books(self):
        """Remove duplicate books based on title and author similarity"""
        print("Deduplicating books...")
        
        if self.books_df is None:
            return
        
        # Create a clean version for deduplication
        books_clean = self.books_df.copy()
        
        try:
            # Clean titles and authors
            books_clean['clean_title'] = books_clean['Book-Title'].apply(self.clean_book_title)
            books_clean['clean_author'] = books_clean['Book-Author'].fillna('').str.lower().str.strip()
            
            # Simple deduplication - group by clean title and author
            unique_books = books_clean.drop_duplicates(subset=['clean_title', 'clean_author'], keep='first')
            
            print(f"Deduplication complete: {len(books_clean)} -> {len(unique_books)} books")
            print(f"Removed {len(books_clean) - len(unique_books)} duplicates")
            
            self.books_deduplicated = unique_books
            
        except Exception as e:
            print(f"Error during deduplication: {e}")
            print("Using original books data without deduplication")
            self.books_deduplicated = books_clean

    def handle_missing_values(self):
        """Handle missing values in all datasets"""
        print("\nHandling missing values...")
        
        # Books data
        if self.books_df is not None:
            print("Cleaning books data...")
            
            # Clean Year-Of-Publication column
            if 'Year-Of-Publication' in self.books_df.columns:
                print("Cleaning Year-Of-Publication column...")
                self.books_df['Year-Of-Publication'] = self.books_df['Year-Of-Publication'].apply(self.clean_year_publication)
                
                # Fill missing years with median
                median_year = self.books_df['Year-Of-Publication'].median()
                if pd.isna(median_year):
                    median_year = 2000
                self.books_df['Year-Of-Publication'].fillna(median_year, inplace=True)
                self.books_df['Year-Of-Publication'] = self.books_df['Year-Of-Publication'].astype(int)
            
            # Fill missing categorical values
            categorical_cols = ['Book-Title', 'Book-Author', 'Publisher']
            for col in categorical_cols:
                if col in self.books_df.columns:
                    missing_count = self.books_df[col].isna().sum()
                    if missing_count > 0:
                        self.books_df[col].fillna('Unknown', inplace=True)
                        print(f"Filled {missing_count} missing values in {col}")
        
        # Ratings data
        if self.ratings_df is not None:
            print("Cleaning ratings data...")
            initial_shape = self.ratings_df.shape
            self.ratings_df.dropna(subset=['User-ID', 'ISBN', 'Book-Rating'], inplace=True)
            print(f"Removed {initial_shape[0] - self.ratings_df.shape[0]} rows with missing values from ratings")
            
            # Ensure Book-Rating is numeric
            self.ratings_df['Book-Rating'] = pd.to_numeric(self.ratings_df['Book-Rating'], errors='coerce')
            self.ratings_df.dropna(subset=['Book-Rating'], inplace=True)
        
        # Users data
        if self.users_df is not None:
            print("Cleaning users data...")
            if 'Age' in self.users_df.columns:
                self.users_df['Age'] = pd.to_numeric(self.users_df['Age'], errors='coerce')
                median_age = self.users_df['Age'].median()
                if pd.isna(median_age):
                    median_age = 30
                self.users_df['Age'].fillna(median_age, inplace=True)
                self.users_df['Age'] = self.users_df['Age'].astype(int)
            
            if 'Location' in self.users_df.columns:
                self.users_df['Location'].fillna('Unknown', inplace=True)
    
    def clean_year_publication(self, year_str):
        """Clean and convert year of publication to numeric"""
        if pd.isna(year_str):
            return np.nan
        
        year_str = str(year_str).strip()
        year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
        if year_match:
            return int(year_match.group())
        return np.nan

    def normalize_data(self):
        """Normalize numerical data with proper zero rating handling"""
        print("Normalizing data...")
        
        if self.ratings_df is not None and 'Book-Rating' in self.ratings_df.columns:
            # Convert to numeric and handle errors
            self.ratings_df['Book-Rating'] = pd.to_numeric(self.ratings_df['Book-Rating'], errors='coerce')
            
            # Remove any NaN values
            initial_count = len(self.ratings_df)
            self.ratings_df = self.ratings_df.dropna(subset=['Book-Rating'])
            print(f"Removed {initial_count - len(self.ratings_df)} invalid ratings")
            
            # Analyze rating distribution
            rating_stats = self.ratings_df['Book-Rating'].describe()
            print("Rating statistics:")
            print(f"  Min: {rating_stats['min']}, Max: {rating_stats['max']}")
            print(f"  Mean: {rating_stats['mean']:.2f}, Std: {rating_stats['std']:.2f}")
            
            # Count zero ratings
            zero_count = (self.ratings_df['Book-Rating'] == 0).sum()
            total_count = len(self.ratings_df)
            
            if zero_count > 0:
                print(f"Found {zero_count} zero ratings ({zero_count/total_count*100:.1f}% of total)")
                print("Removing zero ratings (treating as implicit/unrated)...")
                self.ratings_df = self.ratings_df[self.ratings_df['Book-Rating'] > 0]
            
            # Normalize the remaining ratings to 0-1 scale
            if len(self.ratings_df) > 0:
                current_min = self.ratings_df['Book-Rating'].min()
                current_max = self.ratings_df['Book-Rating'].max()
                
                print(f"Normalizing ratings from {current_min}-{current_max} to 0-1 scale")
                
                if current_max > current_min:
                    self.ratings_df['Book-Rating'] = (self.ratings_df['Book-Rating'] - current_min) / (current_max - current_min)
                else:
                    self.ratings_df['Book-Rating'] = 0.5
                
                final_stats = self.ratings_df['Book-Rating'].describe()
                print(f"Final normalized range: {final_stats['min']:.3f} to {final_stats['max']:.3f}")

    def preprocess_data(self):
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Analyze ratings before processing
        self.analyze_ratings_distribution()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Deduplicate books
        self.deduplicate_books()
        
        # Use deduplicated books for the rest of processing
        if self.books_deduplicated is not None:
            self.books_df = self.books_deduplicated
        
        # Normalize data
        self.normalize_data()
        
        # Filter data for better performance
        self._filter_sparse_data()
        
        # Encode user and book IDs
        self._encode_ids()
        
        print("Data preprocessing completed!")
        print(f"Processed ratings shape: {self.ratings_df.shape}")
        print(f"Processed books shape: {self.books_df.shape}")
        
        return self.ratings_df, self.books_df
    
    def _filter_sparse_data(self):
        """Filter out users and books with too few ratings"""
        if self.ratings_df is None or self.ratings_df.empty:
            return
        
        print("Filtering sparse data...")
        initial_shape = self.ratings_df.shape
        
        # Filter users with at least 3 ratings
        user_rating_counts = self.ratings_df['User-ID'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= 3].index
        self.ratings_df = self.ratings_df[self.ratings_df['User-ID'].isin(active_users)]
        
        # Filter books with at least 3 ratings
        book_rating_counts = self.ratings_df['ISBN'].value_counts()
        popular_books = book_rating_counts[book_rating_counts >= 3].index
        self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(popular_books)]
        
        print(f"Data filtered: {initial_shape[0]} -> {self.ratings_df.shape[0]} ratings")
        print(f"Unique users: {self.ratings_df['User-ID'].nunique()}")
        print(f"Unique books: {self.ratings_df['ISBN'].nunique()}")
    
    def _encode_ids(self):
        """Encode user and book IDs to sequential integers"""
        if self.ratings_df is not None and not self.ratings_df.empty:
            # Create mapping dictionaries
            self.user_ids = self.ratings_df['User-ID'].unique()
            self.book_ids = self.ratings_df['ISBN'].unique()
            
            self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
            self.book_to_idx = {book_id: idx for idx, book_id in enumerate(self.book_ids)}
            self.idx_to_book = {idx: book_id for book_id, idx in self.book_to_idx.items()}
            
            # Apply encoding
            self.ratings_df['user_id_encoded'] = self.ratings_df['User-ID'].map(self.user_to_idx)
            self.ratings_df['book_id_encoded'] = self.ratings_df['ISBN'].map(self.book_to_idx)
            
            print(f"Encoded {len(self.user_to_idx)} users and {len(self.book_to_idx)} books")
    
    def get_surprise_data(self):
        """Convert data to Surprise format"""
        if self.ratings_df is None or self.ratings_df.empty:
            print("No ratings data available for Surprise format")
            return None
            
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(
            self.ratings_df[['user_id_encoded', 'book_id_encoded', 'Book-Rating']], 
            reader
        )
        print("Surprise data prepared successfully")
        return data
    
    def get_book_info(self, book_ids):
        """Get book information for given ISBNs"""
        if self.books_df is None:
            return None
            
        return self.books_df[self.books_df['ISBN'].isin(book_ids)]
    
    def get_popular_books(self, n=50):
        """Get popular books for initial selection (deduplicated)"""
        if self.books_df is None or self.ratings_df is None:
            return pd.DataFrame()
        
        # Calculate average rating and number of ratings for each book
        book_stats = self.ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['count', 'mean']
        }).round(2)
        book_stats.columns = ['rating_count', 'average_rating']
        
        # Merge with book information
        popular_books = self.books_df.merge(book_stats, on='ISBN', how='inner')
        
        # Filter books with sufficient ratings and sort by popularity
        popular_books = popular_books[popular_books['rating_count'] >= 3]
        popular_books = popular_books.sort_values(['rating_count', 'average_rating'], ascending=False)
        
        return popular_books.head(n)
    
    def search_books_deduplicated(self, query, max_results=20):
        """Search books with deduplication applied"""
        if self.books_df is None:
            return pd.DataFrame()
        
        query = query.lower().strip()
        
        # Search in title and author
        mask = (
            self.books_df['Book-Title'].str.lower().str.contains(query, na=False) |
            self.books_df['Book-Author'].str.lower().str.contains(query, na=False)
        )
        
        results = self.books_df[mask].head(max_results * 2)
        
        # Apply additional deduplication for search results
        if not results.empty:
            results = self._deduplicate_search_results(results, query)
        
        return results.head(max_results)
    
    def _deduplicate_search_results(self, results, query):
        """Further deduplicate search results using simple method"""
        try:
            # Create clean titles for comparison
            results_clean = results.copy()
            results_clean['clean_title'] = results_clean['Book-Title'].apply(self.clean_book_title)
            
            # Simple deduplication - keep first occurrence of each clean title + author
            unique_results = results_clean.drop_duplicates(
                subset=['clean_title', 'Book-Author'], 
                keep='first'
            )
            
            return unique_results
            
        except Exception as e:
            print(f"Error in search deduplication: {e}")
            return results