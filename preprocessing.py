"""
Bike Demand Preprocessor v2 - Enhanced
Improved preprocessing with feature engineering and optimized handling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class BikePreprocessor:
    """
    Enhanced preprocessing for bike demand prediction.
    
    Improvements:
    1. Feature engineering: cyclical encoding for temporal features
    2. Robust scaling: handles outliers better than StandardScaler
    3. SMOTE with stratified split: maintains class distribution in folds
    4. Outlier detection: flags extreme weather conditions
    5. Feature interaction: time-of-day × temperature patterns
    
    Usage:
        preprocessor = BikePreprocessor(use_robust_scaling=True, use_interactions=True)
        preprocessor.preprocess('bike_data.csv')
    
    Outputs:
        - X_train.csv
        - X_test.csv
        - y_train.csv
        - y_test.csv
        - preprocessing_report.txt
    """
    
    def __init__(self, target='target', use_robust_scaling=True, use_interactions=True):
        self.target = target
        self.use_robust_scaling = use_robust_scaling
        self.use_interactions = use_interactions
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.feature_names = None
        
    def _create_cyclical_features(self, X):
        """Convert cyclical features (hour, day, month) to sin/cos encoding."""
        cyclical_features = {
            'hour_of_day': 24,
            'day_of_week': 7,
            'month': 12
        }
        
        for feature, max_val in cyclical_features.items():
            if feature in X.columns:
                X[f'{feature}_sin'] = np.sin(2 * np.pi * X[feature] / max_val)
                X[f'{feature}_cos'] = np.cos(2 * np.pi * X[feature] / max_val)
                X = X.drop(columns=[feature])
        
        return X
    
    def _add_feature_interactions(self, X):
        """Add polynomial and interaction features for better model performance."""
        # Temperature interactions with time patterns
        if 'temp' in X.columns and 'hour_of_day_sin' in X.columns:
            X['temp_hour_interaction'] = X['temp'] * X['hour_of_day_sin']
        
        # Humidity × windspeed (affects comfort)
        if 'humidity' in X.columns and 'windspeed' in X.columns:
            X['humidity_wind'] = X['humidity'] * X['windspeed']
        
        # Temperature squared (captures non-linearity: optimal temp range)
        if 'temp' in X.columns:
            X['temp_squared'] = X['temp'] ** 2
        
        # Weather pressure: combines multiple weather features
        if all(c in X.columns for c in ['temp', 'humidity', 'cloudcover']):
            X['weather_index'] = (
                0.3 * (X['temp'] / X['temp'].max()) + 
                0.3 * (1 - X['humidity'] / 100) +  # Low humidity is better
                0.4 * (1 - X['cloudcover'] / 100)  # Clear days are better
            )
        
        return X
    
    def _handle_outliers(self, X):
        """Detect and cap extreme outliers."""
        outlier_threshold = 3  # Standard deviations
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            mean = X[col].mean()
            std = X[col].std()
            upper = mean + outlier_threshold * std
            lower = mean - outlier_threshold * std
            X[col] = X[col].clip(lower, upper)
        
        return X
    
    def preprocess(self, input_file, output_dir='.'):
        """Preprocess data with enhanced feature engineering."""
        
        print("=" * 70)
        print("PREPROCESSING PIPELINE (v2 - Enhanced)")
        print("=" * 70)
        
        # Load
        df = pd.read_csv(input_file)
        print(f"\n1. Loaded data: {df.shape}")
        
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        # Drop low-variance features
        X = X.drop(columns=['dew', 'snow'], errors='ignore')
        print(f"2. Dropped low-variance features (dew, snow): {X.shape}")
        
        # Handle outliers
        X = self._handle_outliers(X)
        print(f"3. Handled outliers with 3σ clipping")
        
        # One-hot encode binary/categorical features
        binary_cats = ['holiday', 'weekday', 'summertime']
        binary_cats = [c for c in binary_cats if c in X.columns]
        X = pd.get_dummies(X, columns=binary_cats, drop_first=False)
        print(f"4. One-hot encoded binary features: {X.shape}")
        
        # Cyclical encoding for temporal features
        X = self._create_cyclical_features(X)
        print(f"5. Applied cyclical encoding (sin/cos for hour, day, month): {X.shape}")
        
        # Add feature interactions
        if self.use_interactions:
            X = self._add_feature_interactions(X)
            print(f"6. Added interaction & polynomial features: {X.shape}")
        else:
            print(f"6. Skipped interaction features")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"\n7. Train/Test split (80/20 stratified):")
        print(f"   - Train: {X_train.shape}, class dist: {y_train.value_counts().to_dict()}")
        print(f"   - Test:  {X_test.shape}, class dist: {y_test.value_counts().to_dict()}")
        
        # SMOTE for class balance (only on training)
        print(f"\n8. Applying SMOTE on training set...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   - After SMOTE: {X_train.shape}, class dist: {y_train.value_counts().to_dict()}")
        
        # Scale features (fit on train, transform on test)
        print(f"\n9. Scaling with {'RobustScaler' if self.use_robust_scaling else 'StandardScaler'}...")
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        print(f"   - Train mean: {X_train.mean().mean():.4f}, std: {X_train.std().mean():.4f}")
        print(f"   - Test mean:  {X_test.mean().mean():.4f}, std: {X_test.std().mean():.4f}")
        
        # Save
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        print(f"\n10. Saved preprocessed data to {output_dir}/")
        print(f"    ✓ X_train.csv: {X_train.shape}")
        print(f"    ✓ X_test.csv:  {X_test.shape}")
        print(f"    ✓ y_train.csv: {y_train.shape}")
        print(f"    ✓ y_test.csv:  {y_test.shape}")
        
        # Generate report
        self._save_report(output_dir, X_train, X_test, y_train, y_test)
        print("=" * 70)

        # Generate report
        self._save_report(output_dir, X_train, X_test, y_train, y_test)
        print("=" * 70)
    
    def _save_report(self, output_dir, X_train, X_test, y_train, y_test):
        """Generate preprocessing summary report."""
        report = f"""
PREPROCESSING SUMMARY REPORT
============================

DATASET STATISTICS:
  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features
  Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features
  
CLASS BALANCE (Training):
  high_bike_demand: {(y_train == 'high_bike_demand').sum()} ({100*(y_train == 'high_bike_demand').sum()/len(y_train):.1f}%)
  low_bike_demand:  {(y_train == 'low_bike_demand').sum()} ({100*(y_train == 'low_bike_demand').sum()/len(y_train):.1f}%)
  
CLASS BALANCE (Test):
  high_bike_demand: {(y_test == 'high_bike_demand').sum()} ({100*(y_test == 'high_bike_demand').sum()/len(y_test):.1f}%)
  low_bike_demand:  {(y_test == 'low_bike_demand').sum()} ({100*(y_test == 'low_bike_demand').sum()/len(y_test):.1f}%)

FEATURE ENGINEERING APPLIED:
  ✓ Outlier handling (3σ clipping)
  ✓ Cyclical encoding for temporal features (sin/cos)
  ✓ Feature interactions (temperature×hour, humidity×wind, etc.)
  ✓ Non-linear features (temperature²)
  ✓ Weather index (composite weather quality metric)
  ✓ SMOTE for class balancing
  ✓ RobustScaler for robust scaling to outliers

FEATURE STATISTICS:
  Train mean: {X_train.mean().mean():.4f}, std: {X_train.std().mean():.4f}
  Test mean:  {X_test.mean().mean():.4f}, std: {X_test.std().mean():.4f}
"""
        
        with open(f'{output_dir}/preprocessing_report.txt', 'w') as f:
            f.write(report)


if __name__ == "__main__":
    # Version 1 (v1 - Basic)
    # preprocessor = BikePreprocessor(target='increase_stock', use_robust_scaling=False, use_interactions=False)
    
    # Version 2 (v2 - Enhanced, RECOMMENDED)
    preprocessor = BikePreprocessor(target='increase_stock', use_robust_scaling=True, use_interactions=True)
    preprocessor.preprocess('/Users/suhrudjoshi/coding/SMLProject/training_data_ht2025.csv', output_dir='.')