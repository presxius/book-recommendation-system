# utils.py
import pickle
import json
import numpy as np
from datetime import datetime

def save_object(obj, filename):
    """Save object to file using pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename):
    """Load object from file using pickle"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def format_metrics(metrics):
    """Format metrics for display"""
    if not metrics:
        return {}
        
    formatted = {}
    for key, value in metrics.items():
        if value is None:
            formatted[key] = "N/A"
        elif isinstance(value, (int, np.integer)):
            formatted[key] = f"{value}"
        elif isinstance(value, float):
            if 'time' in key:
                if 'avg_prediction_time' in key:
                    formatted[key] = f"{value:.3f} ms"
                else:
                    formatted[key] = f"{value:.2f} seconds"
            elif key in ['rmse', 'mae', 'mse']:
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"{value:.2f}"
        elif key == 'last_updated':
            # Format timestamp lebih readable
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                formatted[key] = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted[key] = value
        else:
            formatted[key] = str(value)
    return formatted

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)