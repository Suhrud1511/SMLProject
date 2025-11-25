"""
Bike Demand Preprocessor - Multi-Model Pipeline

Usage:
    from preprocessing import BikePreprocessor
    
    preprocessor = BikePreprocessor(
        target='increase_stock',
        input_file='training_data_ht2025.csv',
        output_dir='.'
    )
    
    preprocessor.tree()        # For Decision Tree, Random Forest
    preprocessor.lda()         # For Linear Discriminant Analysis
    preprocessor.logreg()      # For Logistic Regression
    preprocessor.knn()         # For K-Nearest Neighbors

Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv

Each method applies model-specific preprocessing:
- tree(): No scaling, no encoding (raw features)
- lda(): Box-Cox transformation, one-hot encoding, StandardScaler
- logreg(): One-hot encoding, polynomial features, StandardScaler
- knn(): One-hot encoding, StandardScaler (handles scale-sensitive distance)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split


class BikePreprocessor:
    
    def __init__(self, target='target', input_file=None, output_dir='.'):
        self.target = target
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.X = None
        self.y = None
        
    def _load_data(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.input_file)
                self.X = self.df.drop(columns=[self.target]).copy()
                self.y = (self.df[self.target] == 'high_bike_demand').astype(int)
            except FileNotFoundError:
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
            except KeyError:
                raise KeyError(f"Target column '{self.target}' not found in dataset")
    
    def _save_data(self, X_train, X_test, y_train, y_test):
        try:
            X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
            X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
            y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False, header=False)
            y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False, header=False)
            
            print(f"Saved preprocessed data to {self.output_dir}/")
            print(f"  X_train: {X_train.shape}")
            print(f"  X_test:  {X_test.shape}")
            print(f"  y_train: high={y_train.sum()}, low={(1-y_train).sum()}")
            print(f"  y_test:  high={y_test.sum()}, low={(1-y_test).sum()}")
        except IOError as e:
            raise IOError(f"Failed to save preprocessed data: {str(e)}")
    
    def tree(self):
        self._load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        X_train = X_train.drop(columns=['dew'], errors='ignore')
        X_test = X_test.drop(columns=['dew'], errors='ignore')
        
        X_train = X_train.drop(columns=['snowdepth', 'cloudcover'], errors='ignore')
        X_test = X_test.drop(columns=['snowdepth', 'cloudcover'], errors='ignore')
        
        self._save_data(X_train, X_test, y_train, y_test)
    
    def lda(self):
        self._load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        X_train = X_train.drop(columns=['dew'], errors='ignore')
        X_test = X_test.drop(columns=['dew'], errors='ignore')
        
        numerical_cols = ['temp', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']
        numerical_cols = [c for c in numerical_cols if c in X_train.columns]
        
        X_train_num = X_train[numerical_cols].copy()
        X_test_num = X_test[numerical_cols].copy()
        
        X_train_num = X_train_num.clip(lower=X_train_num.quantile(0.01), upper=X_train_num.quantile(0.99), axis=1)
        X_test_num = X_test_num.clip(lower=X_train_num.quantile(0.01), upper=X_train_num.quantile(0.99), axis=1)
        
        pt = PowerTransformer(method='yeo-johnson')
        X_train_num = pd.DataFrame(
            pt.fit_transform(X_train_num),
            columns=numerical_cols,
            index=X_train.index
        )
        X_test_num = pd.DataFrame(
            pt.transform(X_test_num),
            columns=numerical_cols,
            index=X_test.index
        )
        
        X_train[numerical_cols] = X_train_num
        X_test[numerical_cols] = X_test_num
        
        cats = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
        cats = [c for c in cats if c in X_train.columns]
        X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        self._save_data(X_train, X_test, y_train, y_test)
    
    def logreg(self):
        self._load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        X_train = X_train.drop(columns=['dew'], errors='ignore')
        X_test = X_test.drop(columns=['dew'], errors='ignore')
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        X_train['temp_squared'] = X_train['temp'] ** 2
        X_test['temp_squared'] = X_test['temp'] ** 2
        
        cats = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
        cats = [c for c in cats if c in X_train.columns]
        X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        self._save_data(X_train, X_test, y_train, y_test)
    
    def knn(self):
        self._load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        X_train = X_train.drop(columns=['dew'], errors='ignore')
        X_test = X_test.drop(columns=['dew'], errors='ignore')
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        q1, q99 = X_train.quantile([0.01, 0.99])
        X_train = X_train.clip(lower=q1, upper=q99, axis=1)
        X_test = X_test.clip(lower=q1, upper=q99, axis=1)
        
        cats = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
        cats = [c for c in cats if c in X_train.columns]
        X_train = pd.get_dummies(X_train, columns=cats, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=cats, drop_first=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        self._save_data(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    input_path = '/Users/suhrudjoshi/coding/SMLProject/training_data_ht2025.csv'
    
    preprocessor = BikePreprocessor(target='increase_stock', input_file=input_path, output_dir='.')
    preprocessor.tree()