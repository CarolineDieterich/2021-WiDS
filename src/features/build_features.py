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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

import lightgbm as lgbm

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

    def fill_in_default_na(self):
        self.train_data["gender"] = self.train_data["gender"].fillna("Unknown")
        self.train_data["ethnicity"] = self.train_data["gender"].fillna("Other/Unknown")

        self.test_data["gender"] = self.test_data["gender"].fillna("Unknown")
        self.test_data["ethnicity"] = self.test_data["gender"].fillna("Other/Unknown")
        
    def generate_range_and_mean_from_labs_and_vitals(self):
        vitals = self.data_dictionary[self.data_dictionary["Category"]=="vitals"]["Variable Name"]
        labs = self.data_dictionary[self.data_dictionary["Category"]=="labs"]["Variable Name"]
        
        for v in vitals:
            v = v[3:-4]
            self.train_data[f'd1_{v}_range'] = self.train_data[f'd1_{v}_max'] - self.train_data[f'd1_{v}_min']
            self.train_data[f'd1_{v}_mean'] = (self.train_data[f'd1_{v}_max'] + self.train_data[f'd1_{v}_min']) / 2
        
            self.test_data[f'd1_{v}_range'] = self.test_data[f'd1_{v}_max'] - self.test_data[f'd1_{v}_min']
            self.test_data[f'd1_{v}_mean'] = (self.test_data[f'd1_{v}_max'] + self.test_data[f'd1_{v}_min']) / 2
            
        for v in labs:
            v = v[3:-4]
            self.train_data[f'd1_{v}_range'] = self.train_data[f'd1_{v}_max'] - self.train_data[f'd1_{v}_min']
            self.train_data[f'd1_{v}_mean'] = (self.train_data[f'd1_{v}_max'] + self.train_data[f'd1_{v}_min']) / 2
        
            self.test_data[f'd1_{v}_range'] = self.test_data[f'd1_{v}_max'] - self.test_data[f'd1_{v}_min']
            self.test_data[f'd1_{v}_mean'] = (self.test_data[f'd1_{v}_max'] + self.test_data[f'd1_{v}_min']) / 2

        
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
        
    def merge_apache_and_d_1_max(self):
        'Merge values from Apache Score and d_1_max as they measure the same.'  
        apache_cols = self.train_data.columns[self.train_data.columns.str.contains('apache')]
        apache_cols = [c.split('_apache')[0] for c in apache_cols] 

        vital_cols = self.train_data.columns[self.train_data.columns.str.startswith('d1') & self.train_data.columns.str.contains('_max')]
        vital_cols = [(c.split('d1_')[1]).split('_max')[0] for c in vital_cols]

        common_cols = [c for c in apache_cols if c in vital_cols]
        for c in common_cols:
            if c not in ['resprate', 'temp']:
                # Fill empty d1_..._max column from available ..._apache column
                self.train_data[f"d1_{c}_max"] = np.where((self.train_data[f"d1_{c}_max"].isna() 
                                                    & self.train_data[f"{c}_apache"].notna()), 
                                                   self.train_data[f"{c}_apache"], 
                                                   self.train_data[f"d1_{c}_max"])
                
                self.test_data[f"d1_{c}_max"] = np.where((self.test_data[f"d1_{c}_max"].isna() 
                                                   & self.test_data[f"{c}_apache"].notna()), 
                                                   self.test_data[f"{c}_apache"], 
                                                   self.test_data[f"d1_{c}_max"])
                # Drop ..._apache column
                self.train_data.drop(f"{c}_apache", axis=1, inplace=True)
                self.test_data.drop(f"{c}_apache", axis=1, inplace=True)
    
            
    def generate_features_indicating_that_na(self):
        vitals = self.data_dictionary[self.data_dictionary["Category"]=="vitals"]["Variable Name"]
        labs = self.data_dictionary[self.data_dictionary["Category"]=="labs"]["Variable Name"]
        
        for v in vitals:
            self.train_data[v+"_na"] =self.train_data[v].isna()
            self.test_data[v+"_na"]  =self.test_data[v].isna()
        
        for l in labs:
            self.train_data[l+"_na"] =self.train_data[l].isna()
            self.test_data[l+"_na"]  =self.test_data[l].isna()
        
        #Track where height and weight were missing
        self.train_data["height_na"] = self.train_data["height"].isna() 
        self.test_data["height_na"] = self.test_data["height"].isna() 
        self.train_data["weight_na"] = self.train_data["weight"].isna()  
        self.test_data["weight_na"] = self.test_data["weight"].isna()  

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
        
    def remove_features_with_zero_importance_in_small_LGBM(self):
        train_x = self.train_data.drop("diabetes_mellitus", axis = 1)
        train_y = self.train_data["diabetes_mellitus"]
        params = {
                  'bagging_fraction': 1,
                  'bagging_freq': 40,
                  'bagging_seed': 11,
                  'boosting_type': 'gbdt',  # rf, dart, gbdt
                  'colsample_bytree': 0.2,
                  'device_type': 'cpu',
                  'feature_fraction': 0.5,
                  'lambda_l1': 0,
                  'lambda_l2': 0,
                  'learning_rate': 0.01,
                  'max_bin': 50,
                  'max_depth': 7,  # NOTE auf jeden Fall für Gridsearch verwenden
                  'metric': 'auc',
                  'min_child_samples': 20,
                  'min_data_in_leaf': 100,  # NOTE auf jeden Fall für Gridsearch verwenden
                  'min_data_in_bin': 3,
                  'num_leaves': 80,  # NOTE auf jeden Fall für Gridsearch verwenden (in docs schauen --> max_depth abhängigkeit)
                  'objective': 'binary',
                  'reg_alpha': 0.2,
                  'reg_lambda': 1,
                  'seed': 42,
                  }
        model = lgbm.LGBMClassifier()
        model.set_params(**params)
        
        train_features, valid_features, train_labels, valid_labels = train_test_split(train_x, 
                                                                              train_y, 
                                                                              test_size = 0.15, 
                                                                              stratify=train_y)

        # Train the model with early stopping
        model.fit(train_features, train_labels, eval_metric = 'auc',
                                  eval_set = [(valid_features, valid_labels)],
                                  early_stopping_rounds = 100, verbose = -1)

        feature_importance_values = model.feature_importances_ 

        feature_names = list(train_x.columns)

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
        
        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        
        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])
        self.train_data.drop(to_drop, axis = 1, inplace = True)
        self.test_data.drop(to_drop, axis = 1, inplace = True)
        
    def save_preprocessed_files(self):
        self.train_data.to_csv(Path(self.output_dir + '/train_data.csv'), sep=';')
        self.test_data.to_csv(Path(self.output_dir + '/test_data.csv'), sep=';')

    # FIXME check order
    def run(self):
        # Drop invalid and unnecessary
        self.drop_unnecessary_cols(['Unnamed: 0',
                                    'encounter_id',
                                    'hospital_id',
                                    'icu_id',
                                      'readmission_status'])
           
        self.remove_invalid_rows()

        #Feature Imputation
        # self.add_flag_measurements_first_hour()
        self.create_bmi_categories()
        self.count_comorbidites()
        self.merge_apache_and_d_1_max()
        self.fill_in_default_na()
        self.negative_to_na()
        self.generate_features_indicating_that_na()
        self.generate_range_and_mean_from_labs_and_vitals()
        
        # #Prepare data basics
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

        self.remove_cols_with_nans(missing_perc=0.8)
        self.remove_cols_colinear(correlation_thresh=0.95)
        
        self.train_data = self.create_shuffled_features(self.train_data)
        self.test_data = self.create_shuffled_features(self.test_data)
        
        self.remove_features_with_zero_importance_in_small_LGBM()
        
        #featu_int, featu_float, featu_obj = self.impute_features()
        #self.scale_features(featu_float)
        print(len(self.train_data.columns))
        print(len(self.test_data.columns))
        self.save_preprocessed_files()


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    data_preparer = DataPreparation(input_path=base_path + '/data/raw',
                                    output_dir=base_path + '/data/prepared_data')
    data_preparer.run()
