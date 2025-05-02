"""
Feature Encoding Module
============================

This module provides functionality for encoding categorical variables in pandas DataFrames 
as part of a modular, extensible data preprocessing pipeline. It includes multiple encoding 
strategies for converting non-numeric features into formats suitable for machine learning models.

Core Features:
--------------
1. **Label Encoding**
   - Encodes each category with a unique integer.
   - Appends new columns to the DataFrame with the suffix '_encoded'.

2. **One-Hot Encoding**
   - Expands each categorical feature into multiple binary columns.
   - Columns are automatically named descriptively to avoid ambiguity.

3. **Hashing Encoding**
   - Applies the hashing trick to handle high-cardinality features efficiently.
   - Dynamically determines the optimal number of output dimensions based on the number of unique categories.
   - Issues a warning if the number of unique categories is very small (less than 10), where hashing may not be ideal.

4. **Pipeline Integration**
   - Offers the `EncodeFeatureStep` class, which implements the `PipelineStep` interface for 
     seamless incorporation into machine learning pipelines.

Enums:
------
- `FeatureEncodingMethod`: Enumeration of available encoding strategies:  LABEL_ENCODING, ONEHOT_ENCODING, HASHING

Functions:
----------
- `encode_feature`: Main utility function for encoding categorical columns using the chosen method.
  It automatically selects columns if a subset is not provided.

Classes:
--------
- `EncodeFeatureStep`: A `PipelineStep` implementation for applying feature encoding 
  systematically within data pipelines.
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
    Enumeration of available feature encoding methods.

    Attributes
    ----------
    LABEL_ENCODING : int
        Encodes categories as sequential integers.
    
    ONEHOT_ENCODING : int
        Expands each category into a binary vector.
    
    HASHING : int
        Encodes categories into fixed-length hash-based vectors.
    """
    LABEL_ENCODING = 1
    ONEHOT_ENCODING = 2
    HASHING = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    """
    Helper function to validate and select columns for encoding.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.

    columns_subset : list of str, optional
        Specific columns to consider for encoding. Must be non-numeric.
        If None, all non-numeric (categorical) columns are automatically selected.

    Returns
    -------
    List
        A list of validated column names suitable for encoding.

    Raises
    ------
    ValueError
        If specified columns are invalid (e.g., contain numeric types).
    """
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.

    feature_encoding_method : FeatureEncodingMethod
        The strategy for encoding the features (LABEL_ENCODING, ONEHOT_ENCODING, or HASHING).

    columns_subset : list of str, optional
        Columns to encode. If None, automatically selects all non-numeric columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus newly added encoded columns.

    Raises
    ------
    ValueError
        If NaN values are present in the columns to be encoded.
        If the selected subset includes numeric columns.

    Notes
    -----
    - Encoded columns are **appended** rather than replacing the originals.
    - Label Encoding and One-Hot Encoding generate intuitive, human-readable columns.
    - Hashing Encoding dynamically sets the output dimensions based on the number of categories.
    - For small numbers of categories (<10), Hashing Encoding is discouraged and a warning is emitted.
    """
    # Check if column_subset is valid
    observing_columns = _get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return data

    if data[observing_columns].isna().sum().sum() > 0:
        raise ValueError("Feature encoding does not work when the data contains NaN values.")

    result = data.copy()

    # Ignore warnings
    with warnings.catch_warnings(action="ignore"):
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
                    # Concat the new dataframe to end of original columns
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
                    encoded_columns.columns = ["_".join([col,str(col_gen_name).replace("col_","hash_")]) for col_gen_name in encoded_columns.columns]
                    # Concat the new datafram to end of original columns
                    result = pd.concat([result, encoded_columns], axis=1)

    return result


class EncodeFeatureStep(_PipelineStep):
    """
    Pipeline-compatible step for encoding categorical variables.

    This class allows applying a selected encoding method (LABEL_ENCODING, ONEHOT_ENCODING, or HASHING) to 
    a DataFrame either entirely or to specific columns.

    Parameters
    ----------
    feature_encoding_method : FeatureEncodingMethod
        The method to use for feature encoding.
    
    columns_subset : list of str, optional
        Subset of column names to encode. If None, all categorical columns will be used.
    
    verbose : bool, optional
        If True, it may print some details about the process.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame:
        Applies the selected encoding method to the provided data and returns the transformed DataFrame.

    Raises
    ------
    ValueError
        If the DataFrame has missing values in selected columns or invalid column types.
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