import numpy as np
from surprise import SVD
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
from surprise import accuracy
import pickle
import os

class SVDAutoRec:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = None
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.is_trained = False
        
    def train(self, data, test_size=0.2):
        """Train SVD model with cross-validation to prevent overfitting"""
        print("Training SVD model...")
        
        # Split data
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        
        # Initialize model with regularization to prevent overfitting
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        
        # Train model
        self.model.fit(trainset)
        
        # Evaluate on test set
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"Model trained! RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        self.is_trained = True
        
        return rmse, mae
    
    def hyperparameter_tuning(self, data):
        """Perform grid search for optimal hyperparameters"""
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [20, 30],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.04, 0.06]
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)
        
        # Get best parameters
        best_params = gs.best_params['rmse']
        print(f"Best parameters: {best_params}")
        
        # Update model with best parameters
        self.n_factors = best_params['n_factors']
        self.n_epochs = best_params['n_epochs']
        self.lr_all = best_params['lr_all']
        self.reg_all = best_params['reg_all']
        
        return best_params
    
    def predict_rating(self, user_id, book_id):
        """Predict rating for a user-book pair"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(user_id, book_id).est
    
    def get_top_n_recommendations(self, user_id, books_df, n=10):
        """Get top N recommendations for a user"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Get all book IDs
        book_ids = books_df['book_id_encoded'].unique()
        
        # Predict ratings for all books
        predictions = []
        for book_id in book_ids:
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return predictions[:n]
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")