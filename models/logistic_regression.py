# Logistic Regression on Bike Demand Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

# load preprocessed data
X_train = pd.read_csv('./X_train.csv')
X_test = pd.read_csv('./X_test.csv')
y_train = pd.read_csv('./y_train.csv', header=None).squeeze()
y_test = pd.read_csv('./y_test.csv', header=None).squeeze()

print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# fit logistic regression
lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr.fit(X_train, y_train)

# predictions and probabilities
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

# evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Model Performance ---")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")
print(f"ROC AUC  : {roc_auc:.3f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low','High'], yticklabels=['Low','High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Demand', 'High Demand']))

# ROC curve
RocCurveDisplay.from_estimator(lr, X_test, y_test)
plt.title('ROC Curve - Logistic Regression')
plt.show()

# coefficients and interpretation
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_[0],
    'Odds_Ratio': np.exp(lr.coef_[0])
}).sort_values(by='Odds_Ratio', ascending=False)

print("\nFeature Coefficients and Odds Ratios:")
print(coef_df)

# top 5 features driving high bike demand
print("\nTop 5 features driving high bike demand (largest odds ratios):")
print(coef_df.head(5))
