"""
Feature Scaling Module
=======================

This module provides a flexible framework for scaling and normalizing numeric features 
within pandas DataFrames, designed to integrate seamlessly into modular preprocessing pipelines. 
It supports a variety of scaling methods tailored to different data distributions, 
and offers optional L2 normalization to normalize the magnitude of feature vectors.

Core Features:
--------------
1. **Customizable Feature Scaling**
   - Supports multiple scaling techniques: Min-Max Scaling, Z-Score Standardization, and Robust Scaling.
   - Allows column-specific scaling strategies via user-defined scenarios.
   - Handles scaling safely with built-in validation and informative error handling.

2. **Optional L2 Normalization**
   - Applies L2 normalization across all numeric columns after scaling.
   - Useful for machine learning models sensitive to feature vector magnitudes (e.g., SVM, k-NN).

3. **Pipeline Integration**
   - Implements the `ScaleFeatureStep` class that conforms to the `PipelineStep` interface 
     for easy integration into preprocessing workflows.

Enums:
------
- `_ScalingMethod`: Defines the available feature scaling strategies (MINMAX_SCALING, ZSCORE_STANDARDIZATION, or ROBUST_SCALING)

Functions:
----------
- `scale_feature`: Applies user-specified scaling methods to columns and optionally applies L2 normalization.
- `_get_observing_columns`: Internal utility to validate and select numeric columns for scaling.

Classes:
--------
- `ScaleFeatureStep`: Pipeline-compatible step for flexible feature scaling and normalization.
"""
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from enum import Enum
from ..pipeline.pipeline import _PipelineStep


class _ScalingMethod(Enum):
    """
    Enumeration of supported feature scaling methods.

    Attributes
    ----------
        MINMAX_SCALING: int
            Scales features to a given range (default [0, 1]).
        ZSCORE_STANDARDIZATION: int
            Standardizes features by removing the mean and scaling to standard deviation.
        ROBUST_SCALING: int
            Scales features using statistics that are robust to outliers (median and IQR).
    """
    MINMAX_SCALING = 1
    ZSCORE_STANDARDIZATION = 2
    ROBUST_SCALING = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    """
    Selects and validates numeric columns from the provided DataFrame for scaling.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame.
        columns_subset (List): List of column names to be scaled. If None or empty, all numeric columns are selected.

    Returns
    -------
        List: 
            Validated list of numeric columns to be processed.

    Raises
    ------
        ValueError: 
            If the columns_subset contains non-numeric columns or invalid names.
    """
    # Prepare observing columns
    # Strip whitespaces
    if columns_subset: columns_subset = [col.strip() for col in columns_subset]
    try:
        # If columns_subset only has numeric columns is valid
        numeric_columns = data.select_dtypes(include="number").columns
        # If columns_subset is not None and one of its columns does not exist in numeric columns
        if columns_subset and not all(col in numeric_columns for col in columns_subset):
            raise ValueError("The columns subset contains non-numeric columns!")
        else:
            # If there is a valid subset, it is considered as the observing columns otherwise all numeric columns are considered
            observing_columns = columns_subset if columns_subset else numeric_columns
    except:
        raise ValueError("The columns subset is not valid!")

    return observing_columns


def scale_feature(data : pd.DataFrame, scaling_scenario : Dict, apply_l2normalization : bool = False) -> pd.DataFrame:
    """
    Applies user-specified scaling methods to numeric features, with optional L2 normalization.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame containing numeric features.
        scaling_scenario (Dict): A dictionary specifying the columns and corresponding scaling methods.
            Example:
            {
                "column": ["Age", "Income"],
                "scaling_method": ["MINMAX_SCALING", "ROBUST_SCALING"]
            }
            Supported scaling methods: {"MINMAX_SCALING", "ZSCORE_STANDARDIZATION", "ROBUST_SCALING"}.
        apply_l2normalization (bool, optional): If True, applies L2 normalization across all numeric columns after scaling.

    Returns
    -------
        pd.DataFrame
            DataFrame with scaled and optionally normalized features.

    Raises
    -------
        ValueError
            If columns and scaling methods mismatch, invalid methods are provided, or NaN values are detected.
    """
    # Sample scale_scenario:
    # {"column":["High School Percentage", "Age"],
    #  "scaling_method":["ROBUST_SCALING", "ZSCORE_STANDARDIZATION"]}

    if len(scaling_scenario["column"]) != len(scaling_scenario["scaling_method"]):
        raise ValueError("Number of columns and scaling methods do not match!")

    # Check if column_subset is valid
    observing_columns = _get_observing_columns(data, scaling_scenario["column"])
    if len(observing_columns) == 0: return data

    scaling_scenario["scaling_method"] = [sm.strip() for sm in scaling_scenario["scaling_method"]]
    # Check all the provided scaling_method to be valid
    if not all(sm in [c.name for c in list(_ScalingMethod)] for sm in scaling_scenario["scaling_method"]):
        raise ValueError("At least one of the scaling methods provided in the scenario is not valid! The only acceptable data types are: {MINMAX_SCALING, ZSCORE_STANDARDIZATION, ROBUST_SCALING}")

    if data[observing_columns].isna().sum().sum() > 0:
        raise ValueError("Feature scaling does not work when the data contains NaN values.")

    result = data.copy()

    # Create a list of tuples (column, scaling_method)
    scale_scenario_zipped = list(zip(observing_columns,scaling_scenario["scaling_method"]))

    # For each column in the list, we apply the proper scaling method
    # Then update the date[column]
    for column, scaling_method in scale_scenario_zipped:
        match scaling_method:
            case _ScalingMethod.MINMAX_SCALING.name:
                minmax_scaler = MinMaxScaler()
                result[column] = minmax_scaler.fit_transform(result[[column]])
            case _ScalingMethod.ZSCORE_STANDARDIZATION.name:
                zscore_standardization = StandardScaler()
                result[column] = zscore_standardization.fit_transform(result[[column]])
            case _ScalingMethod.ROBUST_SCALING.name:
                robust_scaler = RobustScaler()
                result[column] = robust_scaler.fit_transform(result[[column]])

    # If apply_l2normalization then we apply l2 normalization on all numeric columns of the dataset
    # Since this type of normalization only makes sense if it applies on all numeric columns
    if apply_l2normalization:
        l2_normalizer = Normalizer()
        # Extract all numeric columns
        numeric_columns = result.select_dtypes("number")
        data_transformed = l2_normalizer.fit_transform(numeric_columns)
        col_number = 0
        for col in numeric_columns.columns: 
            # Update dataset based on all numeric columns which are normalized
            result[col] = data_transformed[:,col_number]
            col_number += 1 

    return result


class ScaleFeatureStep(_PipelineStep):
    """
    Pipeline step for applying feature scaling and optional L2 normalization to a DataFrame.

    This class integrates with a modular DataPipeline system, enabling flexible column-wise
    scaling strategies using common methods like MINMAX_SCALING, ZSCORE_STANDARDIZATION, and ROBUST_SCALING.
    L2 normalization can optionally be applied across all numeric columns to standardize feature magnitudes.

    Parameters
    ----------
    scaling_scenario : Dict
        A dictionary defining which columns to scale and the scaling method for each.
        Example:
            {
                "column": ["Age", "Income"],
                "scaling_method": ["MINMAX_SCALING", "ZSCORE_STANDARDIZATION"]
            }

    apply_l2normalization : bool, optional
        If True, performs L2 normalization on all numeric columns after the specified scaling. Default is False.

    verbose : bool, optional
        If True, it may print some details about the process.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame:
        Applies the defined scaling and normalization to the provided DataFrame.
    """
    def __init__(self, 
                scaling_scenario : Dict,
                apply_l2normalization : bool = False,
                verbose : bool = False
                ):
        
        self.scaling_scenario = scaling_scenario
        self.apply_l2normalization = apply_l2normalization
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        return scale_feature(data, self.scaling_scenario, self.apply_l2normalization)