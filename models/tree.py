import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


class TreeClassifierPipeline:

    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path, random_state=42):
        self.X_train = pd.read_csv(X_train_path)
        self.X_test = pd.read_csv(X_test_path)
        self.y_train = pd.read_csv(y_train_path).squeeze()
        self.y_test = pd.read_csv(y_test_path).squeeze()
        
        # Convert string labels to binary (0/1)
        self.y_train = (self.y_train == 'high_bike_demand').astype(int)
        self.y_test = (self.y_test == 'high_bike_demand').astype(int)
        
        self.random_state = random_state
        self.results_dir = 'tree_results'
        os.makedirs(self.results_dir, exist_ok=True)

        self.dt_model = None
        self.rf_model = None
        self.results = {}

    def train_decision_tree(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        self.dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.dt_model.fit(self.X_train, self.y_train)
        return self.dt_model

    def train_random_forest(self, n_estimators=100, max_depth=10, min_samples_leaf=10):
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.random_state
        )
        self.rf_model.fit(self.X_train, self.y_train)
        return self.rf_model

    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }

        self.results[model_name] = metrics
        return metrics




    def run(self):
        print(f"Loaded training data: {self.X_train.shape}")
        print(f"Loaded test data: {self.X_test.shape}")
        print(f"Train target distribution:\n{self.y_train.value_counts()}\n")

        print("Training Decision Tree...")
        self.train_decision_tree(max_depth=10, min_samples_split=20, min_samples_leaf=10)

        print("Training Random Forest...")
        self.train_random_forest(n_estimators=100, max_depth=10, min_samples_leaf=10)

        print("\nEvaluating Decision Tree...")
        dt_metrics = self.evaluate_model(self.dt_model, 'Decision Tree')
        print(f"DT F1-Score: {dt_metrics['f1']:.4f}")

        print("Evaluating Random Forest...")
        rf_metrics = self.evaluate_model(self.rf_model, 'Random Forest')
        print(f"RF F1-Score: {rf_metrics['f1']:.4f}")



if __name__ == '__main__':
    pipeline = TreeClassifierPipeline(
        'X_train.csv', 
        'X_test.csv', 
        'y_train.csv', 
        'y_test.csv'
    )
    pipeline.run()

# #Results after running the script:
# Data loaded: (1600, 17)
# Target distribution:
# target_binary
# 0    1312
# 1     288
# Name: count, dtype: int64

# Preparing data...

# Train: 1280, Test: 320

# Training Decision Tree...
# Training Random Forest...

# Evaluating Decision Tree...
# DT F1-Score: 0.7020
# Evaluating Random Forest...
# RF F1-Score: 0.6759