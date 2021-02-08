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
