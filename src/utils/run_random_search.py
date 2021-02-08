from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path

from modeling_utils import (
    create_cv_stratified_split,
    create_target,
    create_features
)


class RandomSearch():
    def __init__(self, input_path, folds=3):
        self.folds = folds
        self.train_data = pd.read_csv(
            str(input_path) + '/train_data.csv',
            sep=';'
        )
        self.Y = create_target(self.train_data)
        self.X = create_features(self.train_data)
        self.cv = create_cv_stratified_split(self.folds, self.X, self.Y)

    def run(self, classifier, params, param_combinations):
        random_search = RandomizedSearchCV(
            classifier,
            param_distributions=params,
            n_iter=param_combinations,
            scoring='roc_auc',
            n_jobs=4,
            cv=self.cv,
            verbose=3,
            random_state=42
        )
        random_search.fit(self.X, self.Y)

        print('\n All results:')
        print(random_search.cv_results_)
        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)


if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[2])

    classifier = XGBClassifier(
        learning_rate=0.02,
        n_estimators=600,
        objective='binary:logistic'
    )

    # TODO: Load data from config file
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

    param_combinations = 1

    random_search = RandomSearch(input_path=base_path + '/data/prepared_data')
    random_search.run(classifier, params, param_combinations)
