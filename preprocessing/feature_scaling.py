"""
Type Conversion Module
=======================

This module provides robust functionality for converting column datatypes in pandas DataFrames
as part of a modular preprocessing pipeline. It supports both automatic inference and 
user-defined conversion strategies to ensure that data types are clean, consistent, 
and analysis-ready.

Core Features:
--------------
1. **Automatic Datatype Conversion**
   - Converts object-type columns to numeric where possible.
   - If numeric conversion fails, attempts datetime conversion.
   - Leaves columns unchanged if neither is applicable.

2. **User-Defined Datatype Conversion**
   - Allows precise control over datatype transformations via a structured scenario dictionary.
   - Supports "int", "float", and "datetime" types, with optional format strings for datetime.
   - Includes input validation and detailed error messaging for safe conversion.

3. **Pipeline Integration**
   - Implements the `TypeConversionStep` class conforming to the `PipelineStep` interface 
     for streamlined integration into data preprocessing workflows.

Enums:
------
- `ConvertDatatypeMethod`: Enum defining the conversion strategy (`AUTO`, `USER_DEFINED`).

Functions:
----------
- `convert_datatype_auto`: Applies automatic datatype conversion to object-type columns.
- `convert_datatype_userdefined`: Converts specified columns based on user-provided rules and formats.

Classes:
--------
- `TypeConversionStep`: A pipeline-compatible class that applies either automatic or user-defined 
  datatype conversions to a DataFrame.
"""
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from enum import Enum
from ..pipeline.pipeline import _PipelineStep


class _ScalingMethod(Enum):
    """
    Enumeration of supported feature scaling methods.

    Attributes:
        MINMAX_SCALING: Scales features to a given range (default [0, 1]).
        ZSCORE_STANDARDIZATION: Standardizes features by removing the mean and scaling to standard deviation.
        ROBUST_SCALING: Scales features using statistics that are robust to outliers (median and IQR).
    """
    MINMAX_SCALING = 1
    ZSCORE_STANDARDIZATION = 2
    ROBUST_SCALING = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
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
    Applies specified scaling methods to given numeric columns and optionally applies L2 normalization.

    Args:
        data (pd.DataFrame): The input DataFrame.
        scale_scenario (Dict): Dictionary specifying columns and their respective scaling methods.
            Example:
            {
                "column": ["Age", "Income"],
                "scaling_method": ["MINMAX_SCALING", "ZSCORE_STANDARDIZATION"]
            }
        apply_l2normalization (bool): Whether to apply L2 normalization to all numeric columns.

    Returns:
        pd.DataFrame: The DataFrame with scaled and optionally normalized columns.

    Raises:
        ValueError: If column names and scaling methods mismatch or if invalid methods are provided.
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

    # Create a list of tuples (column, scaling_method)
    scale_scenario_zipped = list(zip(observing_columns,scaling_scenario["scaling_method"]))

    # For each column in the list, we apply the proper scaling method
    # Then update the date[column]
    for column, scaling_method in scale_scenario_zipped:
        match scaling_method:
            case _ScalingMethod.MINMAX_SCALING.name:
                minmax_scaler = MinMaxScaler()
                data[column] = minmax_scaler.fit_transform(data[[column]])
            case _ScalingMethod.ZSCORE_STANDARDIZATION.name:
                zscore_standardization = StandardScaler()
                data[column] = zscore_standardization.fit_transform(data[[column]])
            case _ScalingMethod.ROBUST_SCALING.name:
                robust_scaler = RobustScaler()
                data[column] = robust_scaler.fit_transform(data[[column]])

    # If apply_l2normalization then we apply l2 normalization on all numeric columns of the dataset
    # Since this type of normalization only makes sense if it applies on all numeric columns
    if apply_l2normalization:
        l2_normalizer = Normalizer()
        # Extract all numeric columns
        numeric_columns = data.select_dtypes("number")
        data_transformed = l2_normalizer.fit_transform(numeric_columns)
        col_number = 0
        for col in numeric_columns.columns: 
            # Update dataset based on all numeric columns which are normalized
            data[col] = data_transformed[:,col_number]
            col_number += 1 

    return data


class ScaleFeatureStep(_PipelineStep):
    """
    Pipeline step for applying feature scaling and optional L2 normalization.

    This class integrates with the DataPipeline system and enables users to define
    column-specific scaling strategies using common methods such as MinMax, Z-Score,
    and Robust scaling. It also supports optional L2 normalization applied across all
    numeric features after initial scaling.

    Parameters
    ----------
    scaling_scenario : Dict
        A dictionary specifying the columns to be scaled and their associated methods.
        Example:
            {
                "column": ["Age", "Income"],
                "scaling_method": ["MINMAX_SCALING", "ZSCORE_STANDARDIZATION"]
            }
        Accepted methods are: "MINMAX_SCALING", "ZSCORE_STANDARDIZATION", "ROBUST_SCALING".

    apply_l2normalization : bool, optional
        If True, applies L2 normalization across all numeric columns after scaling.

    verbose : bool, optional
        If True, prints summary information before and after scaling (currently not implemented).
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