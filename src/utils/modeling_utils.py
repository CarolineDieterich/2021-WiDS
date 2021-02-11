#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Modeling_utils.py: our own functions that are needed in various modeling scripts
- create features and targets
- create train_test_splits
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def create_target(train_data: pd.DataFrame) -> pd.DataFrame:
    y = train_data['diabetes_mellitus']
    return y


def create_features(train_data: pd.DataFrame) -> pd.DataFrame:
    X = train_data.drop(['diabetes_mellitus'], axis=1)
    return X


def create_train_test_split(features: pd.DataFrame, target: pd.DataFrame,
                            train_perc: float, test_perc: float):

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        train_size=train_perc,
                                                        test_size=test_perc,
                                                        random_state=42,
                                                        stratify=target)
    return X_train, X_test, y_train, y_test


def create_cv_stratified_split(folds, X, Y):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return skf.split(X, Y)


def save_and_export_predictions(predictions: np.array, file_name: str, input_path: str, output_path: str):
    """
    Export predictions to csv format for submission

    Args:
        predictions:

    Returns:

    """

    sample_submission = pd.read_csv(input_path + "/UnlabeledWiDS2021.csv")
    IDs = sample_submission["encounter_id"]

    # merge array of predictions with encounter id again
    # FIXME find better way to ensure that indices match predictions
    to_submit = {'encounter_id': IDs, 'diabetes_mellitus': predictions}
    df_to_submit = pd.DataFrame(to_submit).set_index(['encounter_id'])

    df_to_submit.to_csv(output_path + f"/{file_name}.csv")
    print("Submission ready.")