import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from tpot import TPOTClassifier


def read_and_preprocess_data(file, target, train_perc, test_perc):
    """
    Import data and do v basic preprocessing

    Params:

    Returns:

    """

    data = pd.read_csv(file)

    y = data[target]

    X = data.drop([target], axis=1)

    X.drop("Unnamed: 0", axis = 1, inplace = True)
    X.drop("encounter_id", axis = 1, inplace = True)
    X.drop("hospital_id", axis = 1, inplace = True)
    X.drop("icu_id", axis = 1, inplace = True)
    X.drop("icu_stay_type", axis = 1, inplace = True)

    categorical_cols = ['elective_surgery',
                       'ethnicity',
                       'gender',
                       'hospital_admit_source',
                       'icu_admit_source',
                       'icu_type',
                       'readmission_status']

    X = pd.get_dummies(X, columns=categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(),
                                                        train_size=train_perc, test_size=test_perc,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

def train_tpot_classifier(X_train, X_test, y_train, y_test):
    """
    Generate classifier for tpot

    TODO add more tunable options

    Params:

    Returns:

    """

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

    # define search
    pipeline_optimizer = TPOTClassifier(generations=100, # default 100
                          population_size=100,  # default 100
                          verbosity=2,
                          random_state=42,
                          scoring='roc_auc',
                          cv=cv,
                          warm_start=True,
                          # periodic_checkpoint_folder='/tpot_checkpoints',
                          early_stop=5)

    # perform search
    pipeline_optimizer.fit(X_train, y_train)

    print(pipeline_optimizer.score(X_test, y_test))

    # export the best model
    pipeline_optimizer.export('tpot_digits_pipeline.py')

    return pipeline_optimizer


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = read_and_preprocess_data('../data/TrainingWiDS2021.csv',
                                                                'diabetes_mellitus',
                                                                0.7,
                                                                0.3)

    pipeline_optimizer = train_tpot_classifier(X_train, X_test, y_train, y_test)
