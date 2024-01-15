from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.metrics import confusion_matrix


def binary_classify_based_on_text(
    df: pd.DataFrame,
    text_col_name: str,
    label_col_name: str,
    C: int = 10,
    max_iter: int = 2000,
    test_size: float = 0.4,
    random_state: int = 42,
) -> tuple:
    """
    Predicts the label based on the text column using Logistic Regression.
    Thus, the labels must be binary.

    Parameters
    ----------
    * df (pd.DataFrame): The dataframe containing the text and label columns.
    * text_col_name (str): The name of the column containing the text.
    * label_col_name (str): The name of the column containing the label that we
      want to predict.
    * C (int, optional): The inverse of regularization strength. Smaller values
      specify stronger regularization.
    * max_iter (int, optional): Maximum number of iterations taken for the
      solvers to converge.
    * test_size (float, optional): The proportion (from 0 to 1) of the dataset
      to include in the test split.
    * random_state (int, optional): Seed used by the random number generator.

    Returns
    -------
    * (tuple): A tuple containing the confusion matrix values (tp, fp, fn, tn).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col_name],
        df[label_col_name],
        test_size=test_size,
        random_state=random_state,
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=150)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    print("Information: This function is currently using 'LogisticRegression'. You may want to change this to 'SGDClassifier' depending on the task.")
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[1, 1]
    return tp, fp, fn, tn


def classify_based_on_text(
    df: pd.DataFrame,
    text_col_name: str,
    label_col_name: str,
    C: int = 10,
    max_iter: int = 2000,
    test_size: float = 0.4,
    random_state: int = 42,
) -> tuple:
    """
    Predicts the label based on the text column using Ridge Classifier. Thus,
    the labels must be categories.

    Parameters
    ----------
    * df (pd.DataFrame): The dataframe containing the text and label columns.
    * text_col_name (str): The name of the column containing the text.
    * label_col_name (str): The name of the column containing the label that we
      want to predict.
    * test_size (float, optional): The proportion (from 0 to 1) of the dataset
      to include in the test split.
    * random_state (int, optional): Seed used by the random number generator.

    Returns
    -------
    * (tuple): A tuple containing the confusion matrix and class names,
      allowing to plot the confusion matrix easily.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col_name],
        df[label_col_name],
        test_size=test_size,
        random_state=random_state,
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=150)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm, clf.classes_


def regression_based_on_text(
    df: pd.DataFrame,
    text_col_name: str,
    label_col_name: str,
    test_size: float = 0.4,
    random_state: int = 42,
) -> tuple:
    """
    Predicts the label based on the text column using Linear Regression. Thus,
    the labels must be numeric.

    Parameters
    ----------
    * df (pd.DataFrame): The dataframe containing the text and label columns.
    * text_col_name (str): The name of the column containing the text.
    * label_col_name (str): The name of the column containing the label that we
      want to predict.
    * test_size (float, optional): The proportion (from 0 to 1) of the dataset
      to include in the test split.
    * random_state (int, optional): Seed used by the random number generator.

    Returns
    -------
    * (tuble): y_true, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col_name],
        df[label_col_name],
        test_size=test_size,
        random_state=random_state,
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=150)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred
