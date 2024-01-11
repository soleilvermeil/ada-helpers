import pandas as pd
import numpy as np


def compute_ccdf(values: list | np.ndarray) -> tuple:
    """
    Computes the complementary cumulative distribution function (CCDF) of a
    list of values. The returned tuple contains the sorted values and the
    CCDF. Thus, plotting the CCDF is simply done by defining x = first tuple
    element and y = second tuple element.
    """
    sorted_values = np.sort(values)
    ccdf = 1 - np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, ccdf


def number_of_unique_x_per_y(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Returns a DataFrame with the number of unique values in x_col for each
    value in y_col. The returned DataFrame contains two columns: each value of
    y_col and the number of unique values in x_col for that value of y_col.
    """
    col_name = f"Number of unique {x_col} per {y_col}"
    # return df.groupby(y_col).nunique()[[x_col]].rename(columns={x_col: col_name}).reset_index()
    return (
        df
        .groupby(y_col)[x_col]
        .nunique()
        .reset_index()
        .rename(columns={x_col: col_name})
    )


def fraction_of_x_per_y(df: pd.DataFrame, x_col: str, x_val, y_col: str) -> pd.DataFrame:
    """
    Returns a DataFrame with the fraction of x_val in x_col for each value in
    y_col. The returned DataFrame contains two columns: each value of y_col and
    the fraction of x_val in x_col for that value of y_col.
    """
    col_name = f"Fraction of {x_col}={x_val} per {y_col}"
    result_df = (
        df
        .groupby(y_col)[x_col]
        .apply(lambda x: np.mean(x == x_val))
        .reset_index()
        .rename(columns={x_col: col_name})
    )
    return result_df


def average_number_of_x_per_y1_per_y2(df: pd.DataFrame, x_col: str, y1_col: str, y2_col: str) -> pd.DataFrame:
    """
    Returns a DataFrame with the average number of rows per value in y1_col for
    each value in y2_col. The returned DataFrame contains two columns: each
    value of y2_col and the average number of rows per value in y1_col for
    that value of y2_col.
    """
    col_name = f"Average number of {x_col} per {y1_col} per {y2_col}"
    return (
        df
        .groupby([y1_col, y2_col])[x_col]
        .count()
        .reset_index()
        .groupby(y2_col)[[x_col]]
        .mean()
        .reset_index()
        .rename(columns={x_col: col_name})
    )