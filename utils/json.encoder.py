import json
import numpy as np
import pandas as pd
from datetime import datetime

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy and pandas types"""
    
    def default(self, obj):
        # Handle numpy integers
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        
        # Handle numpy floats
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        
        # Handle pandas DataFrame
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        
        # Handle pandas Timestamp
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        
        # Handle numpy bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle None values from pandas
        elif pd.isna(obj):
            return None
        
        # Let the base class default method raise the TypeError
        return super().default(obj)

def json_dumps_safe(obj, **kwargs):
    """Safe JSON dumps that handles numpy/pandas types"""
    return json.dumps(obj, cls=NumpyJSONEncoder, **kwargs)

def jsonify_safe(data):
    """Safe version of jsonify that handles numpy/pandas types"""
    from flask import jsonify
    return jsonify(json.loads(json_dumps_safe(data)))