import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Load training data, will need to be changed to the correct path when csv is preprocessed  
training_data = pd.read_csv("training_data_ht2025.csv")
print(training_data.info())

# Convert target variable to categorical codes, not necessary if already done in csv
training_data["increase_stock"] = training_data["increase_stock"].astype("category").cat.codes

X = training_data.drop(columns=["increase_stock", "day_of_week"])
y = training_data["increase_stock"]

def discriminant_analysis(model, X, y):
    """
    Performs discriminant analysis using the specified model on the provided feature and target data.
    Splits the data into training and test sets, fits a pipeline with scaling and the model,
    predicts class labels, and prints a confusion matrix and accuracy score.

    Parameters:
        model: scikit-learn discriminant analysis model (e.g., LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
        X: pandas.DataFrame, feature data
        y: pandas.Series or array-like, target labels

    Output:
        Prints the confusion matrix and accuracy score for the test set predictions.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model_da = Pipeline([
        ("scaler", StandardScaler()), # normalize data, not necessary if csv is already normalized
        ("model", model)
    ])
    model_da.fit(X_train, Y_train)
    predict_da = model_da.predict(X_test)

    print(pd.crosstab(predict_da, Y_test))
    print(accuracy_score(Y_test, predict_da))

def discriminant_analysis_kfold(model, X, y):
    """
    Performs discriminant analysis using the specified model on the provided feature and 
    target data with a stratified k-fold cross-validation.
    Splits the data into training and test sets, fits a pipeline with scaling and the model,
    predicts class labels, and prints a confusion matrix and accuracy score.

    Parameters:
        model: scikit-learn discriminant analysis model (e.g., LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
        X: pandas.DataFrame, feature data
        y: pandas.Series or array-like, target labels

    Output:
        Prints the confusion matrix and accuracy score for the test set predictions.
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

        model_da = Pipeline([
            ("scaler", StandardScaler()), # normalize data, not necessary if csv is already normalized
            ("model", model)
        ])
        model_da.fit(X_train, Y_train)
        predict_da = model_da.predict(X_test)

        print(pd.crosstab(predict_da, Y_test))
        print(accuracy_score(Y_test, predict_da))


#  run discriminant analysis with LDA and QDA
print("LDA")
LDA = discriminant_analysis(skl_da.LinearDiscriminantAnalysis(), X, y)
LDA = discriminant_analysis_kfold(skl_da.LinearDiscriminantAnalysis(), X, y)
print("QDA")
QDA = discriminant_analysis(skl_da.QuadraticDiscriminantAnalysis(reg_param=0.1), X, y)
QDA = discriminant_analysis_kfold(skl_da.QuadraticDiscriminantAnalysis(reg_param=0.1), X, y)