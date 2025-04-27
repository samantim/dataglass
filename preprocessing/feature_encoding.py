"""
Feature Encoding Module
============================

This module provides functionality for encoding categorical variables in pandas DataFrames 
as part of a modular data preprocessing pipeline. It includes multiple strategies 
for encoding non-numeric features to prepare data for machine learning algorithms.

Core Features:
--------------
1. **Label Encoding**
   - Encodes categorical values as integers using `LabelEncoder` from scikit-learn.
   - Appends new encoded columns to the DataFrame with the suffix `_encoded`.

2. **One-Hot Encoding**
   - Converts categorical variables into binary vectors using `OneHotEncoder` from scikit-learn.
   - Creates new columns for each category with descriptive names.

3. **Hashing Encoding**
   - Uses the hashing trick to encode high-cardinality categorical features via `HashingEncoder` from the `category_encoders` library.
   - Automatically calculates the number of components based on the number of unique categories to balance between performance and collisions.
   - Emits a warning when used on low-cardinality columns (less than 10 categories).

4. **Pipeline Integration**
   - Includes the `EncodeFeatureStep` class, which implements the `PipelineStep` interface for seamless integration 
     into preprocessing pipelines.

Enums:
------
- `FeatureEncodingMethod`: Defines strategies for encoding categorical values, including label, one-hot, and hashing encoding.

Functions:
----------
- `encode_feature`: Encodes categorical columns using the specified encoding strategy. Automatically determines 
  applicable columns or uses a provided subset. Encoded columns are appended to the original DataFrame.

Classes:
--------
- `EncodeFeatureStep`: A class implementing the `PipelineStep` interface for encoding categorical variables within a data pipeline.
"""
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
from enum import Enum
from math import log2, ceil
import warnings
from ..pipeline.pipeline import _PipelineStep

class FeatureEncodingMethod(Enum):
    """
    Enumeration of feature encoding methods.

    Options:
        LABEL_ENCODING (int): Encodes categories as integers.
        ONEHOT_ENCODING (int): Encodes categories as one-hot vectors.
        HASHING (int): Encodes categories using the hashing trick.
    """
    LABEL_ENCODING = 1
    ONEHOT_ENCODING = 2
    HASHING = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    # Prepare observing columns

    # All categorical columns of the dataset
    categorical_columns = data.select_dtypes(exclude="number").columns.to_list()

    if columns_subset: 
        # Strip whitespaces
        columns_subset = [col.strip() for col in columns_subset]
        try:
            # Check if one of its columns does not exist in categorical columns
            if not all(col in categorical_columns for col in columns_subset):
                raise ValueError("The columns subset contains numeric columns!")
            else:
                # If there is a valid subset, it is considered as the observing columns
                observing_columns = columns_subset
        except:
            raise ValueError("The columns subset is not valid!")
    else:
        # if there is and empty or none columns_subset, all categorical columns will be considered as the observing columns
        observing_columns = categorical_columns

    return observing_columns 


def encode_feature(data : pd.DataFrame, feature_encoding_method : FeatureEncodingMethod, columns_subset : List = None) -> pd.DataFrame:
    """
    Encodes categorical columns using the specified encoding method.

    Args:
        data (pd.DataFrame): Input DataFrame.
        feature_encoding_method (FeatureEncodingMethod): Encoding method to apply. Options: LABEL_ENCODING, ONEHOT_ENCODING, HASHING.
        columns_subset (List, optional): List of categorical columns to encode. If None, all non-numeric columns are used.

    Returns:
        pd.DataFrame: DataFrame with new encoded columns appended.
    """
    # Check if column_subset is valid
    observing_columns = _get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return data

    if data[observing_columns].isna().sum().sum() > 0:
        raise ValueError("Feature encoding does not work when the data contains NaN values.")

    result = data.copy()

    # Ignore warnings
    warnings.simplefilter("ignore")

    # with warnings.catch_warnings(action="ignore"):
    match feature_encoding_method:
        # Encode the observing columns using label encoder
        case FeatureEncodingMethod.LABEL_ENCODING:
            label_encoder = LabelEncoder()
            for col in observing_columns:
                # The names of the new columns are defined and concat them to end of original columns
                result["_".join([col, "encoded"])] = label_encoder.fit_transform(result[col])

        case FeatureEncodingMethod.ONEHOT_ENCODING:
            # Encode observing columns using one-hot encoder
            onehot_encoder = OneHotEncoder(sparse_output=False)
            for col in observing_columns:
                # Create the encoded contents
                encoded_columns = onehot_encoder.fit_transform(result[[col]])
                # The column names also created by the model
                encoded_columns_name = onehot_encoder.get_feature_names_out([col])
                # Create a dataframe with contents and the column names
                encoded_df = pd.DataFrame(data = encoded_columns, columns=encoded_columns_name, index=result.index)
                # Concat the new datafram to end of original columns
                result = pd.concat([result, encoded_df], axis=1)

        case FeatureEncodingMethod.HASHING:
            for col in observing_columns:
                # Find the log2 of the number of categories (number of unique values in the columns)
                # It is the optimal number of components to reduce the collisions while having good performance
                unique_len = len(result[col].unique())
                # Throw a warning for the less number of unique values
                if unique_len < 10: 
                    print(f"Warning: Hashing for category number less than 10 is not reasonable (column='{col}', category number={unique_len}), and the results would not be promising!")
                n_components=ceil(log2(unique_len))
                # Encode the observing columns using hashing encoder
                hashing = ce.HashingEncoder(n_components = n_components, return_df=True)
                # Create the encoded contents
                encoded_columns = hashing.fit_transform(result[[col]])
                # Enhance the names for more clarification
                encoded_columns.columns = ["_".join([col,col_gen_name]) for col_gen_name in encoded_columns.columns]
                # Concat the new datafram to end of original columns
                result = pd.concat([result, encoded_columns], axis=1)

    return result


class EncodeFeatureStep(_PipelineStep):
    """
    Pipeline step for encoding categorical variables using various encoding strategies.

    This class integrates with the DataPipeline system and allows users to apply a selected 
    encoding method to categorical columns. It supports applying the encoding either to all 
    categorical columns or a specified subset.

    Parameters
    ----------
    feature_encoding_method : FeatureEncodingMethod
        Strategy to use for encoding categorical features (e.g., OneHot, Ordinal, Binary).
    
    columns_subset : List, optional
        A list of column names to apply encoding to. If None, the encoding is applied to all 
        categorical columns detected in the DataFrame.

    verbose : bool, optional
        If True, prints details about the encoding process.
    """
    def __init__(self, 
                feature_encoding_method : FeatureEncodingMethod,
                columns_subset : List = None,
                verbose : bool = False
                ):
        
        self.feature_encoding_method = feature_encoding_method
        self.columns_subset = columns_subset
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        return encode_feature(data, self.feature_encoding_method, self.columns_subset)