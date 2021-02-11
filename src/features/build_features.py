#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Prepare_data.py: Currently basic cleaning and preprocessing to get the data into modelling format
# FIXME add more advanced feature engineering steps to improve model performance
"""

import typing as th
from pathlib import Path

import pandas as pd
import numpy as np


class DataPreparation:

    def __init__(self, input_path=None, output_dir=None):
        # Required parameters
        self.input_path = input_path
        self.output_dir = output_dir
        self.train_data = pd.read_csv(str(self.input_path) + '/TrainingWiDS2021.csv')
        self.test_data = pd.read_csv(str(self.input_path) + '/UnlabeledWiDS2021.csv')
        self.data_dictionary = pd.read_csv(str(self.input_path) + '/DataDictionaryWiDS2021.csv')

    def drop_unnecessary_cols(self, cols_to_drop: th.List) -> None:
        self.train_data.drop(cols_to_drop, axis=1, inplace=True)
        self.test_data.drop(cols_to_drop, axis=1, inplace=True)

    def create_dummy_cols(self, categorical_cols: th.List) -> None:
        self.train_data = pd.get_dummies(self.train_data, columns=categorical_cols)
        self.test_data = pd.get_dummies(self.test_data, columns=categorical_cols)

    # FIXME check if this is really necessary or if there is another way to deal (maybe use sth like (list(set)))
    def drop_cols_not_in_test(self, cols_not_in_test: th.List) -> None:
        """Cols that are only in the train set should be dropped for shape reasons"""
        self.train_data.drop(cols_not_in_test, axis=1, inplace=True)

    def remove_invalid_rows(self) -> None:
        """Remove rows where age is 0"""
        self.train_data = self.train_data[self.train_data .age >= 16].reset_index(drop=True)
        self.test_data = self.test_data[self.test_data.age >= 16].reset_index(drop=True)

    def remove_cols_with_nans(self, missing_perc: float) -> None:
        """Remove cols with over xx% missing data"""
        cols_to_keep = self.train_data.columns[self.train_data.isnull().mean() < missing_perc]
        cols_to_keep_test = cols_to_keep.drop('diabetes_mellitus')
        self.train_data = self.train_data[cols_to_keep]
        self.test_data = self.test_data[cols_to_keep_test]

    def remove_cols_colinear(self, correlation_thresh) -> None:
        """Remove highly colinear columns"""
        corr_matrix = self.train_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than thresh
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_thresh)]
        # Drop features
        self.train_data.drop(to_drop, axis=1, inplace=True)
        self.test_data.drop(to_drop, axis=1, inplace=True)

    # FIXME decide whether flag is useful feature or not
    def add_flag_measurements_first_hour(self) -> None:
        """
        If a measurement was done within first hour, set flag
        If any of cols that start with h1 not empty, add flag
        """
        first_hour_cols = self.train_data.filter(like='h1').columns
        # Columns which have at least one value measure within first hour
        first_hour_measurement_present_rows = self.train_data.loc[self.train_data[first_hour_cols].isnull().sum(axis=1)
                                                                  /len(first_hour_cols)!=1]

    # FIXME add further functions for feature engineering here
    def further_feature_engineering(self):
        pass

    def save_preprocessed_files(self):
        self.train_data.to_csv(Path(self.output_dir + '/train_data.csv'), sep=';')
        self.test_data.to_csv(Path(self.output_dir + '/test_data.csv'), sep=';')

    def run(self):
        self.drop_unnecessary_cols(['Unnamed: 0',
                                    'encounter_id',
                                    'hospital_id',
                                    'icu_id',
                                    'icu_stay_type',
                                     'readmission_status'])
        self.create_dummy_cols(['elective_surgery',
                                'ethnicity',
                                'gender',
                                'hospital_admit_source',
                                'icu_admit_source',
                                'icu_type'])
        self.drop_cols_not_in_test(['hospital_admit_source_ICU',
                                    'hospital_admit_source_Other',
                                    'hospital_admit_source_PACU',
                                    'hospital_admit_source_Observation',
                                    'hospital_admit_source_Acute Care/Floor'])
        self.remove_invalid_rows()
        # self.add_flag_measurements_first_hour()
        self.remove_cols_with_nans(missing_perc=0.8)
        self.remove_cols_colinear(correlation_thresh=0.95)
        self.save_preprocessed_files()


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    data_preparer = DataPreparation(input_path=base_path + '/data/raw',
                                    output_dir=base_path + '/data/prepared_data')
    data_preparer.run()
