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

        # workaround for import error on column names, otherwise invalid json
        self.train_data = self.train_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        self.test_data = self.test_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        self.X = create_features(self.train_data)
        self.y = create_target(self.train_data)
        self.X_train, self.X_test, self.y_train, self.y_test = create_train_test_split(self.X,
                                                                                       self.y,
                                                                                       0.7,
                                                                                       0.3)
        self.d_train = lgbm.Dataset(self.X_train, self.y_train, free_raw_data=False)
        self.d_test = lgbm.Dataset(self.X_test, self.y_test, free_raw_data=False)
        self.d_train_full = lgbm.Dataset(self.X, self.y, free_raw_data=False)

    @staticmethod
    # TODO parameter tuning https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    def set_params():
        params = {
                  'bagging_fraction': 1,
                  'bagging_freq': 40,
                  'bagging_seed': 11,
                  'boosting_type': 'gbdt',  # rf, dart, gbdt
                  #'categorical_feature': "",
                  'colsample_bytree': 0.2,
                  'device_type': 'cpu',
                  'feature_fraction': 0.5,
                  # 'is_unbalance': False,
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
                  # 'num_threads': 2,
                  # 'num_iterations': 5000,
                  'objective': 'binary',
                  'reg_alpha': 0.2,
                  'reg_lambda': 1,
                  'seed': 42,
                  }
        return params

    def train_lgbm_cv_with_early_stop(self, params):
        """

        Args:
            params:

        Returns:

        """
        logging.info("Starting lgbm training with early stop and cv...")

        lgbm_cv_hist = lgbm.cv(
            params,
            self.d_train,
            num_boost_round=5000,
            verbose_eval=100,
            early_stopping_rounds=100)

        logging.info("No more improvements found, early stopping...")
        return lgbm_cv_hist

    def train_lgbm(self, params, lgbm_cv_hist):
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
            len(lgbm_cv_hist[list(lgbm_cv_hist.keys())[0]]),
            verbose_eval=100)

        return lgbm_model

    def make_preds_test(self, lgbm_model) -> np.array:
        """

        Args:
            lgbm_model:

        Returns:

        """
        logging.info("Calculating lgbm predictions...")
        predictions_test = lgbm_model.predict(self.X_test)
        print(roc_auc_score(self.y_test, predictions_test))
        return predictions_test

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
        print(feature_importances[0:30])

    def train_final_model(self, lgbm_model, params):
        """

        Args:
            lgbm_model:
            params:

        Returns:

        """
        logging.info("Final training with hyperparameters on full dataset...")
        # training without early stop
        lgbm_trainall = lgbm.train(params,
                                    self.d_train_full,
                                    lgbm_model.current_iteration(),
                                    verbose_eval=100)
        return lgbm_trainall

    def make_and_save_final_preds(self, lgbm_trainall, file_name):
        """

        Args:
            lgbm_trainall:
            file_name:

        Returns:

        """
        preds_unlabeled_lgbm = lgbm_trainall.predict(self.test_data)
        save_and_export_predictions(preds_unlabeled_lgbm, file_name, self.raw_path, self.output_path)
        return preds_unlabeled_lgbm

    def run(self):
        params = self.set_params()
        lgbm_cv_hist = self.train_lgbm_cv_with_early_stop(params)
        lgbm_model = self.train_lgbm(params, lgbm_cv_hist)
        self.get_most_important_features(lgbm_model)
        lgbm_trainall = self.train_final_model(lgbm_model, params)
        predictions_test_set = self.make_preds_test(lgbm_model)
        predictions_val_set = self.make_and_save_final_preds(lgbm_trainall, 'lgbm_preds')
        print(predictions_val_set)


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    lgbm_predictor = LgbmModel(input_path=base_path + '/data/prepared_data',
                               output_path=base_path + '/data/submission',
                               raw_path=base_path + '/data/raw')
    lgbm_predictor.run()

