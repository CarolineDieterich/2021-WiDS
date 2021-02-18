#! /usr/bin/env python

from optparse import OptionParser
from xgboost import XGBClassifier
from pathlib import Path
import sys
import json
from datetime import datetime
from random_search import (
    RandomSearch
)


def parseOpts(args):
    """
    definition an parsing of the command line options
    """
    parser = OptionParser()
    parser.add_option("--params_file", action="store", type="string", dest="params_file")  # noqa
    parser.add_option("--combinations", action="store", type="int", dest="combinations")  # noqa
    parser.add_option("--folds", action="store", type="int", dest="folds")  # noqa

    (options, args) = parser.parse_args(args)
    return options


def start_xgboost_random_search(folds, params_file, combinations):
    base_path = str(Path(__file__).resolve().parents[2])

    # initialize classifier
    classifier = XGBClassifier(
        learning_rate=0.02,
        n_estimators=600,
        objective='binary:logistic'
    )

    # load parameters
    with open(base_path + '/data/params/' + params_file) as f:
        params = json.load(f)

    # add path to preprocessed data
    random_search = RandomSearch(input_path=base_path + '/data/prepared_data', folds)  # noqa

    # start random search
    result = random_search.run(classifier, params, combinations)

    # write results to file
    with open(base_path + '/data/params/result_' + datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + str(combinations) + '_' + params_file, 'w') as f:  # noqa
        json.dump(result, f)


if __name__ == '__main__':
    args = sys.argv[1:]
    o = parseOpts(args)
    start_xgboost_random_search(o.folds, o.params_file, o.combinations)
