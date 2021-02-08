from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pandas as pd

from modeling_utils import create_cv_stratified_split, create_train_test_split, create_target, create_features

# TODO: Restructure
def start_cv_randomsearch(params, classifier, cv_data, X, Y, param_combinations):
    random_search = RandomizedSearchCV(
        classifier,
        param_distributions=params,
        n_iter=param_combinations,
        scoring='roc_auc',
        n_jobs=4,
        cv=cv_data,
        verbose=3,
        random_state=42
    )
    random_search.fit(X, Y)

    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)


if __name__ == "__main__":
    folds = 3
    train_data = pd.read_csv('./data/prepared_data/train_data.csv', sep=';')

    Y = create_target(train_data)
    X = create_features(train_data)

    cv_data = create_cv_stratified_split(folds, X, Y)

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

    param_combinations = 5
    start_cv_randomsearch(params, classifier, cv_data, X, Y, param_combinations)

