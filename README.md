# 2021-WiDS

Repo for WiDS datathon 2021, consisting of data preparation and modeling scripts as well as helper tools, notebooks, and more.

# Get started
Clone this repo

# Prepare data
- `python3 build_features.py`
- Prepared_data will be saved in `data/prepared_data/` and can be used directly for modeling
- Latest version should always be run before starting a new modeling run in order to ensure that the latest preprocessed data is used 

# Modeling
- Helper functions, e.g. to create features and targets and train test split are in `src/utils/modeling_utils.py` 
- Prepared data can be used for modeling

# Random Search
1. Define the parameter option in a json file in data/params/
2. Call the script with:
- the file name
- the number of parameter combinations that should be tested
- and the number of cross-validation splits
```
python src/utils/run_random_search_xgboost.py --folds 2 --params_file params_xgboost.json --combinations 1
```
