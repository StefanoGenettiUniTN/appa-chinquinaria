"""
Module: Data Splitter
Splits the dataset into training and test sets and defines temporal windows.
"""

import pandas as pd

def split_train_test(df, training_start_date, training_end_date, testing_start_date, testing_end_date):
    train_df = df[(df['data'] >= training_start_date) & (df['data'] < training_end_date)]
    test_df = df[(df['data'] >= testing_start_date) & (df['data'] < testing_end_date)]
    return train_df, test_df

def create_time_windows(test_df, window_size_months):
    """
    Create sequential time windows of N months from the test dataset.
    Returns a list of DataFrames (one per window).
    """
    windows = []
    start_date = test_df['data'].min()
    end_date = test_df['data'].max()

    current_start = start_date
    while current_start < end_date:
        current_end = current_start + pd.DateOffset(months=window_size_months)
        window = test_df[(test_df['data'] >= current_start) &
                         (test_df['data'] < current_end)]
        if not window.empty:
            windows.append(window)
        current_start = current_end
    return windows
