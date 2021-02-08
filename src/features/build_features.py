#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Prepare_data.py: Currently basic cleaning and preprocessing to get the data into modelling format
# FIXME add more advanced feature engineering steps to improve model performance
"""

import typing as th
from pathlib import Path

import pandas as pd


class DataPreparation:

    def __init__(self, input_path=None, output_dir=None):
        # Required parameters
        self.input_path = input_path
        self.output_dir = output_dir
        self.train_data = pd.read_csv(str(self.input_path) + '/TrainingWiDS2021.csv')
        self.test_data = pd.read_csv(str(self.input_path) + '/UnlabeledWiDS2021.csv')
        self.data_dictionary = pd.read_csv(str(self.input_path) + '/DataDictionaryWiDS2021.csv')

    def drop_unnecessary_cols(self) -> None:
        self.train_data.drop("Unnamed: 0", axis=1, inplace=True)
        self.train_data.drop("encounter_id", axis=1, inplace=True)
        self.train_data.drop("hospital_id", axis=1, inplace=True)
        self.train_data.drop("icu_id", axis=1, inplace=True)
        self.train_data.drop("icu_stay_type", axis=1, inplace=True)
        self.test_data.drop("Unnamed: 0", axis=1, inplace=True)
        self.test_data.drop("encounter_id", axis=1, inplace=True)
        self.test_data.drop("hospital_id", axis=1, inplace=True)
        self.test_data.drop("icu_id", axis=1, inplace=True)
        self.test_data.drop("icu_stay_type", axis=1, inplace=True)

    def create_dummy_cols(self, categorical_cols: th.List) -> None:
        self.train_data = pd.get_dummies(self.train_data, columns=categorical_cols)
        self.test_data = pd.get_dummies(self.test_data, columns=categorical_cols)

    # FIXME add further functions for feature engineering here
    def further_feature_engineering(self):
        pass

    def save_preprocessed_files(self):
        self.train_data.to_csv(Path(self.output_dir + '/train_data.csv'), sep=';')
        self.test_data.to_csv(Path(self.output_dir + '/test_data.csv'), sep=';')

    def run(self):
        self.drop_unnecessary_cols()
        self.create_dummy_cols(['elective_surgery',
                                'ethnicity',
                                'gender',
                                'hospital_admit_source',
                                'icu_admit_source',
                                'icu_type',
                                'readmission_status'])
        self.save_preprocessed_files()


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    data_preparer = DataPreparation(input_path=base_path + '/data/raw',
                                    output_dir=base_path + '/data/prepared_data')
    data_preparer.run()

