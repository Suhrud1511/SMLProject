from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

training_data = pd.read_csv("training_data_ht2025.csv")

# Convert target variable to categorical codes, not necessary if already done in csv
training_data["increase_stock"] = training_data["increase_stock"].astype("category").cat.codes

X = training_data.drop(columns=["increase_stock"])
y = training_data["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# predicts the most frequent class in the training set
most_frequent_class = Y_train.mode()[0]
y_pred = [most_frequent_class] * len(Y_test)
print(pd.crosstab(y_pred, Y_test))
print(accuracy_score(Y_test, y_pred))