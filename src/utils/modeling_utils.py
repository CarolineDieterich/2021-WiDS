#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Modeling_utils.py: our own functions that are needed in various modeling scripts
- create features and targets
- create train_test_splits
"""

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
                                                        stratify=True)
    return X_train, X_test, y_train, y_test


def create_cv_stratified_split(folds, X, Y):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return skf.split(X, Y)
