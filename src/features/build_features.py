#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Prepare_data.py: Currently basic cleaning and preprocessing to get the data into modelling format
# FIXME add more advanced feature engineering steps to improve model performance
"""

import bisect
import typing as th
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer


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

    # FIXME ensure that this is only done for numerical cols
    def remove_cols_colinear(self, correlation_thresh) -> None:
        """Remove highly colinear columns"""
        corr_matrix = self.train_data.corr(method='pearson').abs()
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

    def create_bmi_categories(self):
        optimal_bmi_range_min_ages = [25, 35, 45, 55, 65]
        optimal_bmi_ranges = [(19, 24), (20, 25), (21, 26), (22, 27), (23, 28), (24, 29)]

        bmi_categories = ["underweight", "normal weight", "overweight", "obesity", "strong obesity"]

        def bmi_ok(data):
            bmi_ok = []
            for idx, row in data.iterrows():
                age = row['age']
                bmi = row['bmi']
                # gender = np.where()
                optimal_bmi_range = optimal_bmi_ranges[bisect.bisect_left(optimal_bmi_range_min_ages, age)]

                if bmi >= optimal_bmi_range[0] and bmi <= optimal_bmi_range[1]:
                    bmi_ok.append(1)
                else:
                    bmi_ok.append(0)
            return np.asarray(bmi_ok)

        def bmi_classes(data):
            bmi_classes = []
            for idx, row in data.iterrows():
                age = row['age']
                bmi = row['bmi']

                if row['gender'] == 'M':  # 1
                    bmi_cat_thresholds = [20, 26, 31, 41]

                elif row['gender'] == 'F':  # 0
                    bmi_cat_thresholds = [19, 25, 31, 41]

                bmi_category = bmi_categories[bisect.bisect_left(bmi_cat_thresholds, bmi)]
                bmi_classes.append(bmi_category)
            return np.asarray(bmi_classes)

        # bmi_ok_array = np.asarray(bmi_ok)
        # bmi_classes_array = np.asarray(bmi_classes)
        self.train_data['bmi_ok'] = bmi_ok(self.train_data)
        self.train_data['bmi_classes_array'] = bmi_classes(self.train_data)
        self.test_data['bmi_ok'] = bmi_ok(self.test_data)
        self.test_data['bmi_classes_array'] = bmi_classes(self.test_data)


    # TODO try mice etc
    def impute_features(self):

        featu_int = []
        featu_float = []
        featu_obj = []

        for col in self.test_data.columns:
            x = self.test_data[col].dtype
            if x == 'int64' or x == 'int32':
                # self.train_data[col] = self.train_data[col].astype('int32')
                featu_int.append(col)
            elif x == 'float64' or x == 'float32':
                # table_train[col] = table_train[col].astype('float32')
                featu_float.append(col)
            else:
                featu_obj.append(col)

        imp_simp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer_int = SimpleImputer(missing_values=np.nan,
                                    strategy='mean')  # KNNImputer(n_neighbors=2, weights="uniform")
        imputer_float = SimpleImputer(missing_values=np.nan,
                                      strategy='mean')  # KNNImputer(n_neighbors=2, weights="uniform")

        # Train
        self.train_data[featu_float] = imputer_float.fit_transform(self.train_data[featu_float])
        self.train_data[featu_int] = imputer_int.fit_transform(self.train_data[featu_int]).astype(int)
        self.train_data[featu_obj] = imp_simp.fit_transform(self.train_data[featu_obj])
        # Test
        self.test_data[featu_float] = imputer_float.transform(self.test_data[featu_float])
        self.test_data[featu_int] = imputer_int.transform(self.test_data[featu_int]).astype(int)
        self.test_data[featu_obj] = imp_simp.transform(self.test_data[featu_obj])

        return featu_int, featu_float, featu_obj

    def scale_features(self, featu_float) -> None:
        qt = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")

        self.train_data[featu_float] = qt.fit_transform(self.train_data[featu_float])
        self.test_data[featu_float] = qt.transform(self.test_data[featu_float])

    def create_shuffled_features(self, data):
        def create_features(x):
            # g_e = x['ethnicity'] + '_' + x['gender']
            a_bmi = x['age'] * x['bmi']
            a_glu = x['age'] * x['d1_glucose_max']
            bmi_glu = x['bmi'] * x['d1_glucose_max']
            return pd.Series([a_bmi, a_glu, bmi_glu], index=['age_bmi', 'a_glu', 'bmi_glu'])
        return data.join(data.apply(create_features, axis=1))

    def negative_to_na(self):
        # code variables that are -1 as na
        num = self.train_data._get_numeric_data()
        num[num < 0] = None

        num_unl = self.test_data._get_numeric_data()
        num_unl[num_unl < 0] = None

    def count_comorbidites(self):
        comorbidities_count_list = [
            'aids',
            'cirrhosis',
            'hepatic_failure',
            'immunosuppression',
            'leukemia',
            'lymphoma',
            'solid_tumor_with_metastasis'
        ]

        self.train_data['comorbities_count'] = self.train_data[comorbidities_count_list].sum(axis=1)
        self.test_data['comorbities_count'] = self.test_data[comorbidities_count_list].sum(axis=1)

    def save_preprocessed_files(self):
        self.train_data.to_csv(Path(self.output_dir + '/train_data.csv'), sep=';')
        self.test_data.to_csv(Path(self.output_dir + '/test_data.csv'), sep=';')

    # FIXME check order
    def run(self):
        self.drop_unnecessary_cols(['Unnamed: 0',
                                    'encounter_id',
                                    'hospital_id',
                                    'icu_id',
                                     'readmission_status'])
        self.remove_invalid_rows()
        # self.add_flag_measurements_first_hour()
        self.train_data = self.create_shuffled_features(self.train_data)
        self.test_data = self.create_shuffled_features(self.test_data)
        self.create_bmi_categories()

        self.create_dummy_cols(['elective_surgery',
                                'ethnicity',
                                'gender',
                                'hospital_admit_source',
                                'icu_admit_source',
                                'icu_type',
                                'icu_stay_type',
                                'bmi_ok',
                                'bmi_classes_array'])

        self.drop_cols_not_in_test(['hospital_admit_source_ICU',
                                    'hospital_admit_source_Other',
                                    'hospital_admit_source_PACU',
                                    'hospital_admit_source_Observation',
                                    'hospital_admit_source_Acute Care/Floor'])

        self.negative_to_na()
        self.count_comorbidites()

        self.remove_cols_with_nans(missing_perc=0.8)
        self.remove_cols_colinear(correlation_thresh=0.95)

        # featu_int, featu_float, featu_obj = self.impute_features()
        # self.scale_features(featu_float)

        print(self.train_data.columns)
        print(self.test_data.columns)
        self.save_preprocessed_files()


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    data_preparer = DataPreparation(input_path=base_path + '/data/raw',
                                    output_dir=base_path + '/data/prepared_data')
    data_preparer.run()
