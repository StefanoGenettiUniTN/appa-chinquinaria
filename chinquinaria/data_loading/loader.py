"""
Module: Data Loader
Loads and preprocesses the dataset from CSV.
"""

import pandas as pd

def load_data(path):
    """
    Load the dataset from CSV and perform initial cleaning.
    """
    df = pd.read_csv(path)
    return df