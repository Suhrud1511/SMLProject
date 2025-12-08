# -*- coding: utf-8 -*-
import os
# from google.colab import files
# uploaded = files.upload()

# -*- coding: utf-8 -*-
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from preprocessing import BikePreprocessor
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# run preprocessing once
preprocessor = BikePreprocessor(
    target='increase_stock',
    input_file=os.path.join(BASE_DIR, 'training_data_ht2025.csv'),
    output_dir=BASE_DIR,
)
preprocessor.logreg()   # not pre.logreg()

#loading the preprocessed data
X_train = pd.read_csv(os.path.join(BASE_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, 'y_train.csv'), header=None).squeeze()
y_test  = pd.read_csv(os.path.join(BASE_DIR, 'y_test.csv'), header=None).squeeze()

#defining the hyperparameter grid and CV strategy
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier()

grid = GridSearchCV(
    knn,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)


grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
print("Best params:", grid.best_params_)

y_pred = best_knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "High"],
            yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()

k_values = [3, 5, 7, 9, 11]
f1_scores = []
