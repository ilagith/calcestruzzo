import pandas as pd
import numpy as np
import miceforest as mice
from miceforest import mean_match_default
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple
pd.set_option('display.max_columns', None)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load csv file ignoring the first two columns, and return a pd.DataFrame.
    :param filename: 'dataset.csv'
    :return data: pd.DataFrame without first two cols (duplicated ids)
    """
    data = pd.read_csv(
        filepath_or_buffer=filename,
        sep=';',
        index_col=0,
        usecols=lambda col: col != 'Unnamed: 0',
    )

    return data


def convert_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to respect convention (no upper cases, no spaces).
    :param data: Loaded pd.DataFrame
    :return data: DataFrame with convention column names
    """
    conventional_column_names = []
    for column in data.columns:
        conventional_column_names.append(
            re.sub(
                pattern=r'(?<!^)(?=[A-Z])',
                repl='_',
                string=column
            ).lower()
        )

    data.columns = conventional_column_names

    return data


def extract_features(data: pd.DataFrame, features_type: str) -> pd.DataFrame:
    """
    Add extracted features to original dataframe. Here, the user can decide to add
    domain features found on research papers ('water_cement_ratio', 'fly_ash_specific_gravity'),
    or features derived from the data itself () or both.
    :param data: Original DataFrame
    :param features_type: Type of features we would like to extract ['domain', 'derived', 'both']
    :return data: Original dataframe + 'domain' or 'derived' or 'both' features
    """
    types_of_features = {
        'domain': ['water_cement_ratio', 'fly_ash_specific_gravity'],
        'derived': ['coarse_aggr_age', 'fine_aggr_age'],
        'both': ['water_cement_ratio', 'fly_ash_specific_gravity', 'coarse_aggr_age', 'fine_aggr_age']
    }

    formulas = {
        'water_cement_ratio': data['water_component'] / data['cement_component'],
        'fly_ash_specific_gravity': data['fly_ash_component'] / data['water_component'],
        'coarse_aggr_age': data['coarse_aggregate_component'] / data['age_in_days'],
        'fine_aggr_age': data['fine_aggregate_component'] / data['age_in_days']
    }

    features_to_add = types_of_features.get(features_type)

    for new_feature in features_to_add:
        data[new_feature] = formulas.get(new_feature)

    return data


def feature_target_split(data: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate target variable from features.
    :param data: DataFrame to separate features and target for
    :param target_name: Column name of target column
    :return
        features: DataFrame of features
        target_var: DataFrame with target variable only
    """
    features = data.loc[:, data.columns != target_name]
    target_var = data[target_name]

    return features, target_var


def impute_na_and_rescale(training_features: np.ndarray, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sklearn pipeline to:
    - Impute missing values using mice
    - Rescale features space
    :param training_features: Array of training features (X_train)
    :param test_features: Array of testing features (X_test)
    :return: Complete and rescaled training (complete_tr_features) and testing features (complete_tst_features)
    """

    # Train
    mice_kernel = mice.ImputationKernel(
        data=training_features,
        mean_match_scheme=mean_match_default.set_mean_match_candidates(2),
        random_state=42
    )

    skl_pipeline = Pipeline([
        ('impute', mice_kernel),
        ('scaler', StandardScaler()),
    ])

    # Test
    complete_tr_features = skl_pipeline.fit_transform(
        training_features,
        impute__iterations=8
    )

    complete_tst_features = skl_pipeline.transform(test_features)

    return complete_tr_features, complete_tst_features
