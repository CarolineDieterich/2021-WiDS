#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Run_lgbm.py: set params for lgbm model, run & evaluate
"""

import re
import logging
import lightgbm as lgbm
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

from modeling_utils import create_target, create_features, create_train_test_split # , save_and_export_predictions


class LgbmModel:

    def __init__(self, input_path=None):
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

    @staticmethod
    def set_params():
        params = {
                  'bagging_fraction': 1,
                  'bagging_freq': 40,
                  'bagging_seed': 11,
                  'boosting_type': 'gbdt',
                  'colsample_bytree': 0.2,
                  'feature_fraction': 0.5,
                  'learning_rate': 0.01,
                  'max_bin': 40,
                  'max_depth': 15,
                  'metric': 'auc',
                  'min_child_samples': 20,
                  'min_data_in_leaf': 50,
                  'num_leaves': 150,
                  'objective': 'binary',
                  'reg_alpha': 0.2,
                  'reg_lambda': 1,
                  }
        return params

    def train_lgbm_with_early_stop(self, params):
        logging.info("Starting lgbm training with early stop and cv...")

        lgbm1_hist = lgbm.cv(
            params,
            self.d_train,
            num_boost_round=5000,
            verbose_eval=100,
            early_stopping_rounds=100)

        logging.info("No more improvements found, early stopping...")
        return lgbm1_hist

    def train_lgbm(self, params, lgbm1_hist):
        logging.info("Training lgbm model...")

        lgbm1 = lgbm.train(
            params,
            self.d_train,
            len(lgbm1_hist[list(lgbm1_hist.keys())[0]]),
            verbose_eval=100)

        return lgbm1

    def make_preds(self, lgbm1):
        logging.info("Calculating lgbm predictions...")

        preds_lgbm1 = lgbm1.predict(self.X_test)
        print(roc_auc_score(self.y_test, preds_lgbm1))

        return preds_lgbm1

    def run(self):
        params = self.set_params()
        lgbm1_hist = self.train_lgbm_with_early_stop(params)
        lgbm1 = self.train_lgbm(params, lgbm1_hist)
        predictions = self.make_preds(lgbm1)
        # save_and_export_predictions(predictions, self.X_test)


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    lgbm_predictor = LgbmModel(input_path=base_path + '/data/prepared_data')
    lgbm_predictor.run()

