import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_aqi_model(df):
    """
    Trains a pipeline to predict PM2.5 concentrations based on core 
    gaseous emission indicators commonly available in the dataset.
    """
    # Use standard features found in global time-series emissions datasets
    # Mapping typical fallback columns if names vary slightly
    possible_features = ['no2', 'co', 'so2', 'o3', 'pm10']
    features = [col for col in possible_features if col in df.columns]
    
    # Fallback to general numeric values if exact names don't align perfectly
    if len(features) < 2:
        features = list(df.select_dtypes(include=[np.number]).columns)
        if 'pm2_5' in features: features.remove('pm2_5')
        if 'pm2.5' in features: features.remove('pm2.5')
            
    target = 'pm2_5' if 'pm2_5' in df.columns else 'pm2.5'
    
    if target not in df.columns:
        # Create a mock target or select first numeric columns if dataset schema shifts
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        target = numeric_cols[0]
        features = numeric_cols[1:5]

    print(f"Training Model with Features: {features} -> Target: {target}")
    
    # Handle missing records safely
    clean_df = df[features + [target]].dropna()
    
    X = clean_df[features]
    y = clean_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pack scaler and estimator tightly within an isolated Pipeline object
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Track the feature names inside the pipeline configuration for structural reconstruction
    model_pipeline.feature_names_ = features
    model_pipeline.target_name_ = target
    
    # Export to runtime folder directory
    joblib.dump(model_pipeline, 'aqi_model.pkl')
    print("Model successfully trained and saved as 'aqi_model.pkl'")
    
    return model_pipeline, features, target