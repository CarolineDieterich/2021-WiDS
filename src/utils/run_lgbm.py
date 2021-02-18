#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Run_lgbm.py: set params for lgbm model, run & evaluate
"""

import re
import logging
import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

from modeling_utils import create_target, create_features, create_train_test_split, save_and_export_predictions


class LgbmModel:

    def __init__(self, input_path=None, output_path=None, raw_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.raw_path = raw_path

        self.train_data = pd.read_csv(str(input_path) + '/train_data.csv', sep=';')
        self.test_data = pd.read_csv(str(input_path) + '/test_data.csv', sep=';')

        # remove unimportant columns (from model feature importance)
        cols_to_drop = []
        # cols_to_drop = ['aids', 'hospital_admit_source_ICU to SDU', 'ethnicity_Other/Unknown', 'hospital_admit_source_Floor', 'ethnicity_Native American', 'icu_type_CCU-CTICU', 'hospital_admit_source_Operating Room', 'intubated_apache', 'bmi_classes_array_normal weight', 'icu_type_Cardiac ICU', 'icu_type_CTICU', 'icu_stay_type_transfer', 'icu_stay_type_readmit', 'icu_type_Neuro ICU', 'elective_surgery_0', 'bmi_classes_array_strong obesity', 'bmi_classes_array_underweight', 'hepatic_failure', 'ethnicity_Hispanic', 'icu_stay_type_admit', 'hospital_admit_source_Recovery Room', 'icu_admit_source_Other Hospital', 'apache_post_operative', 'hospital_admit_source_Step-Down Unit (SDU)', 'solid_tumor_with_metastasis', 'hospital_admit_source_Chest Pain Center', 'hospital_admit_source_Other Hospital', 'immunosuppression', 'leukemia', 'icu_admit_source_Other ICU', 'hospital_admit_source_Other ICU', 'lymphoma', 'aids', 'hospital_admit_source_ICU to SDU']
        self.train_data.drop(cols_to_drop, axis=1, inplace=True)
        self.test_data.drop(cols_to_drop, axis=1, inplace=True)

        print(self.train_data.shape)
        print(self.test_data.shape)
        print(list(set(self.train_data.columns) - set(self.test_data.columns)))

        # workaround for import error on column names, otherwise invalid json
        self.train_data = self.train_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        self.test_data = self.test_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        self.X = create_features(self.train_data)
        self.y = create_target(self.train_data)
        self.X_train, self.X_test, self.X_valid, self.y_train, \
        self.y_test, self.y_valid = create_train_test_split(self.X, self.y, 0.7, 0.3)

        self.d_train = lgbm.Dataset(self.X_train, self.y_train, free_raw_data=False)
        self.d_test = lgbm.Dataset(self.X_test, self.y_test, free_raw_data=False)
        self.d_train_full = lgbm.Dataset(self.X, self.y, free_raw_data=False)

    @staticmethod
    # TODO parameter tuning https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    def set_params():
        params = {'boosting': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  'learning_rate': 0.005,  # typical values: 0.1, 0.001, 0.003
                  'num_leaves': 90,  # nr of leaves in full tree, default: 31
                  # 'reg_alpha': 0.1,
                  'max_bin': 50,  # max nr of bin that feature value will bucket in
                  'max_depth': 7,  # maximum depth of tree (may lead to overfitting)
                  'min_data_in_leaf': 50,   # minimum number of records a leaf may have (default 20)
                  'min_child_samples': 30,
                  'feature_fraction': 0.5, # this amount of parameters is selected randomly in each iteration for building tree
                  'bagging_fraction': 0.5, #  fraction of data to be used for each iteration - speed up training + avoid overfitting
                  'bagging_freq': 9,
                  'bagging_seed': 11,
                  'lambda_l1': 0.2,  # specifies regularization (typical values from 0 to 1)
                  'lambda_l2': 1,  # specifies regularization (typical values from 0 to 1)
                  'device': 'cpu'
                  }
        return params

    # def train_lgbm_cv_with_early_stop(self, params):
    #     """
    #
    #     Args:
    #         params:
    #
    #     Returns:
    #
    #     """
    #     logging.info("Starting lgbm training with early stop and cv...")
    #
    #     lgbm_cv_hist = lgbm.cv(
    #         params,
    #         self.d_train,
    #         num_boost_round=10000,
    #         verbose_eval=100,
    #         early_stopping_rounds=100)
    #
    #     return lgbm_cv_hist

    def train_lgbm(self, params):
        """

        Args:
            params:
            lgbm_cv_hist:

        Returns:

        """
        logging.info("Training lgbm model...")

        lgbm_model = lgbm.train(
            params,
            self.d_train,
            num_boost_round=10000,
            valid_sets=[self.d_test],
            # evals_result=len(lgbm_cv_hist[list(lgbm_cv_hist.keys())[0]]),
            verbose_eval=100,
            early_stopping_rounds=200)

        return lgbm_model

    def make_preds(self, lgbm_model) -> np.array:
        """

        Args:
            lgbm_model:

        Returns:

        """
        logging.info("Calculating lgbm predictions...")
        predictions_test = lgbm_model.predict(self.X_test)
        print("Test set score : " + str(roc_auc_score(self.y_test, predictions_test)))
        predictions_val = lgbm_model.predict(self.X_valid)
        print("Valid set score : " + str(roc_auc_score(self.y_valid, predictions_val)))


    @staticmethod
    def get_most_important_features(lgbm_model) -> np.array:
        """

        Args:
            lgbm_model:

        Returns:

        """
        logging.info("Retrieving most important features for lgbm model...")
        feature_importance_values = lgbm_model.feature_importance()
        feature_names = lgbm_model.feature_name()
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
            'importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        # Make sure most important features are on top
        feature_importances = feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = feature_importances[feature_importances['cumulative_importance'] > 0.99]

        # to_drop = list(record_low_importance['feature'])

        to_drop.extend(list(record_low_importance['feature']))

        print(feature_importances[0:30])
        print(to_drop)

    # def train_final_model(self, lgbm_model, params):
    #     """
    #
    #     Args:
    #         lgbm_model:
    #         params:
    #
    #     Returns:
    #
    #     """
    #     logging.info("Final training with hyperparameters on full dataset...")
    #     # training without early stop
    #     lgbm_trainall = lgbm.train(params,
    #                                 self.d_train_full,
    #                                 lgbm_model.current_iteration(),
    #                                 verbose_eval=100)
    #     return lgbm_trainall

    def make_and_save_final_preds(self, lgbm_model, file_name):
        """

        Args:
            lgbm_trainall:
            file_name:

        Returns:

        """
        preds_unlabeled_lgbm = lgbm_model.predict(self.test_data)
        save_and_export_predictions(preds_unlabeled_lgbm, file_name, self.raw_path, self.output_path)
        return preds_unlabeled_lgbm

    def run(self):
        params = self.set_params()
        # Train with cv (?)
        # lgbm_cv_hist = self.train_lgbm_cv_with_early_stop(params)
        lgbm_model = self.train_lgbm(params)
        # Print most important features
        self.get_most_important_features(lgbm_model)
        # Print preds
        self.make_preds(lgbm_model)
        # Exclude least important features and train again
        # try again without scaling
        self.make_and_save_final_preds(lgbm_model, 'lgbm_preds_scaled_remove_unimportant_feats')

        # Train again on full dataset (?)


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    lgbm_predictor = LgbmModel(input_path=base_path + '/data/prepared_data',
                               output_path=base_path + '/data/submission',
                               raw_path=base_path + '/data/raw')
    lgbm_predictor.run()

