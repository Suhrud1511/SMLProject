import os
import sys
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import BikePreprocessor

preprocessor = BikePreprocessor(
    target='increase_stock',
    input_file='training_data_ht2025.csv',
    output_dir='.'
)
preprocessor.lda()

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv', header=None).squeeze()
y_test = pd.read_csv('y_test.csv', header=None).squeeze()

print(f"Loaded preprocessed LDA data:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}\n")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}\n")
    
    return {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}

def cross_validate_model(model_class, X, y, n_splits=5, model_name="Model"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []
    roc_aucs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        
        print(f"{model_name} Fold {fold}:")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n{model_name} Cross-Validation Results:")
    print(f"  Mean Accuracy: {pd.Series(accuracies).mean():.4f} ± {pd.Series(accuracies).std():.4f}")
    print(f"  Mean F1-Score: {pd.Series(f1_scores).mean():.4f} ± {pd.Series(f1_scores).std():.4f}")
    print(f"  Mean ROC-AUC: {pd.Series(roc_aucs).mean():.4f} ± {pd.Series(roc_aucs).std():.4f}\n")


if __name__ == '__main__':
    
    print("LINEAR DISCRIMINANT ANALYSIS (LDA)")
    
    print("\nTrain/Test Evaluation:")
    evaluate_model(skl_da.LinearDiscriminantAnalysis(), X_train, X_test, y_train, y_test, "LDA")
    
    print("Cross-Validation Evaluation:")
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    cross_validate_model(skl_da.LinearDiscriminantAnalysis, X_full, y_full, n_splits=5, model_name="LDA")
    
   
    print("QUADRATIC DISCRIMINANT ANALYSIS (QDA)")
    
    print("\nTrain/Test Evaluation:")
    evaluate_model(skl_da.QuadraticDiscriminantAnalysis(), X_train, X_test, y_train, y_test, "QDA")
    
    print("Cross-Validation Evaluation:")
    cross_validate_model(skl_da.QuadraticDiscriminantAnalysis, X_full, y_full, n_splits=5, model_name="QDA")