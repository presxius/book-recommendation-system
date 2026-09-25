# model_training.py
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error
import os
import random

def plot_metrics(metrics, save_path="metrics_plot.png"):
    """
    Visualisasi evaluation metrics sebagai diagram batang.
    Muncul langsung di VSCode terminal (python plot_test.py).
    """
    labels = ['RMSE', 'MAE']
    values = [
        metrics.get('rmse', 0),
        metrics.get('mae', 0),
    ]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}" if isinstance(height, float) else f"{height}",
            ha='center',
            va='bottom'
        )

    plt.title("Model Evaluation Metrics")
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()


class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None
        self.trainset = None
        self.testset = None
        self.metrics = {
            'rmse': 0.0,
            'mae': 0.0, 
            'training_time': 0.0,
            'avg_prediction_time': 0.0,
            'test_samples': 0
        }  # Inisialisasi dengan nilai default
        
    def prepare_surprise_data(self, ratings_df):
        """Prepare data for Surprise library"""
        print("Preparing data for Surprise...")
        
        # Create Surprise reader
        reader = Reader(rating_scale=(1, 10))
        
        # Prepare data for Surprise (user_id, book_id, rating)
        surprise_data = Dataset.load_from_df(
            ratings_df[['user_id_encoded', 'book_id_encoded', 'Book-Rating']], 
            reader
        )
        
        return surprise_data
    
    def train_model(self, ratings_df, test_size=0.2, force_retrain=False):
        """Train SVD model with hyperparameter tuning"""
        model_path = 'models/svd_model.pkl'
        
        # Check if model exists and we don't want to retrain
        if os.path.exists(model_path) and not force_retrain:
            print("Loading existing model...")
            self.model = joblib.load(model_path)
            return self.model
        
        print("Starting model training...")
        start_time = time.time()
        
        # Prepare data
        surprise_data = self.prepare_surprise_data(ratings_df)
        self.trainset, self.testset = train_test_split(surprise_data, test_size=test_size)
        
        # Hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [20, 25],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.1]
        }
        
        print("Grid search in progress...")
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=2, n_jobs=-1)
        gs.fit(surprise_data)
        
        print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
        print(f"Best MAE: {gs.best_score['mae']:.4f}")
        print(f"Best params: {gs.best_params['rmse']}")
        
        # Train final model with best parameters
        best_params = gs.best_params['rmse']
        self.model = SVD(
            n_factors=best_params['n_factors'],
            n_epochs=best_params['n_epochs'],
            lr_all=best_params['lr_all'],
            reg_all=best_params['reg_all']
        )
        
        print("Training final model...")
        self.model.fit(self.trainset)
        
        # Evaluate model - INI YANG MENYIMPAN METRICS
        self.evaluate_model()
        
        # Test prediction speed
        self.test_prediction_speed()
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)
        
        training_time = time.time() - start_time
        self.metrics['training_time'] = training_time
        
        print(f"✅ Model training completed in {training_time:.2f} seconds")
        
        # Print metrics untuk debugging
        print("=== TRAINING METRICS ===")
        for key, value in self.metrics.items():
            print(f"  {key}: {value}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        try:
            # Test set predictions
            test_predictions = self.model.test(self.testset)
            
            # Calculate metrics
            rmse = accuracy.rmse(test_predictions, verbose=False)
            mae = accuracy.mae(test_predictions, verbose=False)
            
            # Extract actual and predicted ratings for additional metrics
            actual_ratings = [pred.r_ui for pred in test_predictions]
            predicted_ratings = [pred.est for pred in test_predictions]
            
            # UPDATE METRICS - PASTIKAN INI DIEKSEKUSI
            self.metrics.update({
                'rmse': float(rmse),
                'mae': float(mae), 
                'test_samples': len(test_predictions)
            })
            
            print(f"📊 Model Evaluation:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Test samples: {len(test_predictions)}")
            
        except Exception as e:
            print(f"❌ Error in model evaluation: {e}")
            # Set default values jika evaluation gagal
            self.metrics.update({
                'rmse': 1.0,
                'mae': 1.0,
                'test_samples': 0
            })
    
    def test_prediction_speed(self, n_predictions=200):
        """Test prediction speed for the model"""
        if not self.model:
            print("❌ No model available for prediction speed test")
            return None
            
        print(f"Testing prediction speed with {n_predictions} predictions...")
        
        try:
            # Use simple range for user and item IDs
            user_ids = list(range(50))
            item_ids = list(range(50))
            
            start_time = time.perf_counter()
            
            # Perform multiple predictions
            for i in range(n_predictions):
                user = random.choice(user_ids)
                item = random.choice(item_ids)
                self.model.predict(user, item)
            
            end_time = time.perf_counter()
            
            # Calculate average prediction time in milliseconds
            total_time_ms = (end_time - start_time) * 1000
            avg_prediction_time = total_time_ms / n_predictions
            
            # UPDATE METRICS
            self.metrics['avg_prediction_time'] = float(avg_prediction_time)
            self.metrics['prediction_test_samples'] = n_predictions
            
            print(f"  Average prediction time: {avg_prediction_time:.4f} ms")
            
            return avg_prediction_time
            
        except Exception as e:
            print(f"❌ Error testing prediction speed: {e}")
            # Set default value jika testing gagal
            self.metrics['avg_prediction_time'] = 1.0
            return 1.0
    
    def get_metrics(self):
        """Return training metrics"""
        return self.metrics