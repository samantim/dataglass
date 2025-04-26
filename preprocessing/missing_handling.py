"""
Missing Values Handling Module
===============================

This module provides functionality for handling missing values in pandas DataFrames 
as part of a modular data preprocessing pipeline. It includes multiple strategies 
for handling missing values, including dropping rows, imputing based on data types, 
and imputing using adjacent values.

Core Features:
--------------
1. **Drop Missing Values**
   - Drops all rows containing missing values from the DataFrame.

2. **Datatype-Based Imputation**
   - Imputes missing values in numeric columns using the specified method (mean, median, or mode).
   - Imputes missing values in categorical columns using the mode (most frequent value).

3. **Adjacent Value Imputation**
   - Imputes missing values based on adjacent values, such as forward fill, backward fill, linear interpolation, 
     or time-based interpolation (when a datetime column is provided).

4. **Pipeline Integration**
   - Includes the `HandleMissingStep` class, which implements the `PipelineStep` interface for seamless integration 
     into preprocessing pipelines.

Enums:
------
- `AdjacentImputationMethod`: Defines imputation methods for filling missing values using adjacent data.
- `NumericDatatypeImputationMethod`: Specifies imputation methods for numeric columns (mean, median, mode).
- `HandleMissingMethod`: Defines strategies for handling missing values (drop, datatype imputation, or adjacent imputation).

Functions:
----------
- `handle_missing_values_drop`: Drops rows with missing values from the DataFrame.
- `handle_missing_values_datatype_imputation`: Imputes missing values based on column data types (numeric or categorical).
- `handle_missing_values_adjacent_value_imputation`: Imputes missing values using adjacent values (forward fill, backward fill, interpolation).

Classes:
--------
- `HandleMissingStep`: A class implementing the `PipelineStep` interface for handling missing values within a data pipeline.
"""
import pandas as pd
from enum import Enum
from ..pipeline.pipeline import _PipelineStep


class AdjacentImputationMethod(Enum):
    """
    Enum for specifying the method to impute missing values based on adjacent data.

    Attributes:
        FORWARD (int): Forward fill - propagates last valid observation forward.
        BACKWARD (int): Backward fill - uses next valid observation to fill missing values.
        INTERPOLATION_LINEAR (int): Linearly interpolates missing values.
        INTERPOLATION_TIME (int): Interpolates missing values assuming the index is time-based.
    """
    # Enum classes make the code cleaner and avoid using invalid inputs
    FORWARD = 1
    BACKWARD = 2
    INTERPOLATION_LINEAR = 3
    INTERPOLATION_TIME = 4


class NumericDatatypeImputationMethod(Enum):
    """
    Enum for specifying imputation methods applicable to numeric columns.

    Attributes:
        MEAN (int): Impute missing values using the mean of the column.
        MEDIAN (int): Impute missing values using the median of the column.
        MODE (int): Impute missing values using the mode of the column (most frequent value).
    """
    # Enum classes make the code cleaner and avoid using invalid inputs
    # This enum specifies the method of imputation only for numeric columns, since for the categorical columns "MODE" is the only option
    MEAN = 1
    MEDIAN = 2
    MODE = 3


class HandleMissingMethod(Enum):
    """
    Enum for specifying the strategy to handle missing values in the dataset.

    Attributes
    ----------
    DROP : int
        Drop rows that contain missing values.
    DATATYPE_IMPUTATION : int
        Impute missing values based on column data types (mean/median/mode for numeric, mode for categorical).
    ADJACENT_VALUE_IMPUTATION : int
        Impute missing values using adjacent values (forward fill, backward fill, or interpolation).
    """
    DROP = 1
    DATATYPE_IMPUTATION = 2
    ADJACENT_VALUE_IMPUTATION = 3


def handle_missing_values_drop(data: pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    """
    Drop all rows in the DataFrame that contain missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with potential missing values.
    verbose : bool, optional
        If True, prints information about the dataset before and after dropping missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing missing values removed.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Drop all rows containing missing values
    data = data.dropna()

    # Check dataset after dropping missing values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def handle_missing_values_datatype_imputation(data : pd.DataFrame, numeric_datatype_imputation_method : NumericDatatypeImputationMethod, verbose : bool = False) -> pd.DataFrame:
    """
    Handle missing values by imputing them based on column data types.
    Numeric columns are imputed using the specified method; non-numeric (categorical) columns use the mode.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with potential missing values.
    numeric_datatype_imputation_method : NumericDatatypeImputationMethod
        Method to use for imputing numeric columns (mean, median, or mode).
    verbose : bool, optional
        If True, prints information about the dataset before and after imputation.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """

    result = data.copy(deep=False)

    # Display dataset info before and after imputation if verbose is enabled
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    if verbose:
        print(f"Dataset has {result.shape[0]} rows before handling missing values.\nMissing values are:\n{result.isna().sum()}")

    for col in result.columns:
        # Check if the columns is numeric
        if pd.api.types.is_numeric_dtype(result[col]):
            match numeric_datatype_imputation_method:
                case NumericDatatypeImputationMethod.MEAN:
                    result[col] = result[col].fillna(result[col].mean())
                case NumericDatatypeImputationMethod.MEDIAN:
                    result[col] = result[col].fillna(result[col].median())
                case NumericDatatypeImputationMethod.MODE:
                    # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
                    result[col] = result[col].fillna(result[col].mode()[0])
                case _:
                    raise ValueError("Numeric datatype imputation method is not valid. It should be selected using NumericDatatypeImputationMethod enum.")
        else:
            # If it is a categorical columns
            # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
            result[col] = result[col].fillna(result[col].mode()[0])

    # Check dataset after dropping missing values
    if verbose:
        print(f"Dataset has {result.shape[0]} rows after handling missing values.")

    return result


def handle_missing_values_adjacent_value_imputation(data: pd.DataFrame, adjacent_imputation_method : AdjacentImputationMethod, time_reference_col : str = "", verbose : bool = False) -> pd.DataFrame:
    """
    Handle missing values using adjacent value techniques such as forward fill, backward fill, or interpolation.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with potential missing values.
    adjacent_imputation_method : AdjacentImputationMethod
        Method to use for adjacent value imputation (forward, backward, linear interpolation, or time-based interpolation).
    time_reference_col : str, optional
        Name of the datetime column required for time-based interpolation.
        Must be in a valid datetime format. Default is an empty string.
    verbose : bool, optional
        If True, prints information about the dataset before and after imputation.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using adjacent value methods.

    Raises
    ------
    ValueError
        If the time reference column is missing or not in a valid datetime format when using time-based interpolation.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Check for missing values
    # It is also possible to use isnull() instead of isna()

    result = data.copy(deep=False)

    if verbose:
        print(f"Dataset has {result.shape[0]} rows before handling missing values.\nMissing values are:\n{result.isna().sum()}")

    # Fill missing values using adjacent value imputation
    match adjacent_imputation_method:
        case AdjacentImputationMethod.FORWARD:
            # Fill missing values using forward fill (Note that fillna(method='ffill') method is deprecated)
            result = result.ffill()

        case AdjacentImputationMethod.BACKWARD:
            # Fill missing values using backward fill (Note that fillna(method='bfill') method is deprecated)
            result = result.bfill()

        case AdjacentImputationMethod.INTERPOLATION_LINEAR:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Fill missing values using linear interpolation (Default method is linear)
            for col in result.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].interpolate(method='linear')

        case AdjacentImputationMethod.INTERPOLATION_TIME:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Check if time reference column is provided, as it is needed for time interpolation
            if not time_reference_col:
                raise ValueError("Time reference column is required for time interpolation.")
            else:
                try:
                    # Convert time reference column to datetime if contains datatime values, otherwise it will raise an error
                    result[time_reference_col] = pd.to_datetime(result[time_reference_col])
                except KeyError:
                    raise KeyError(
                        f"The column '{time_reference_col}' is not a valid column name."
                    )
                except ValueError:
                    raise ValueError(
                        f"The column '{time_reference_col}' is not in a valid datetime format. "
                        f"This method requires a datetime column for time-based interpolation."
                    )
                                
            # Setting the datetime column as index is necessary for time-based interpolation
            result = result.set_index(time_reference_col)

            # Fill missing values using time interpolation
            for col in result.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].interpolate(method='time')
            
            # Reset index to original
            result = result.reset_index()
        case _:
            raise ValueError("Adjacent imputation method is not valid. It should be selected using AdjacentImputationMethod enum.")

    # Check dataset after dropping missing values
    if verbose:
        print(f"Dataset has {result.shape[0]} rows after handling missing values.")

    return result


class HandleMissingStep(_PipelineStep):
    """
    Pipeline step for handling missing values using various strategies.

    This class integrates with the DataPipeline system and allows users to choose 
    how to handle missing values by either:
    - Dropping rows with missing values.
    - Imputing based on data type (mean, median, mode for numerics; mode for categoricals).
    - Using adjacent values (forward fill, backward fill, linear and time-based interpolation).

    Parameters
    ----------
    handle_missing_method : HandleMissingMethod
        Strategy to use for handling missing values.
    numeric_datatype_imputation_method : NumericDatatypeImputationMethod, optional
        Required if `handle_missing_method` is DATATYPE_IMPUTATION. Specifies how to impute numeric columns.
    adjacent_imputation_method : AdjacentImputationMethod, optional
        Required if `handle_missing_method` is ADJACENT_VALUE_IMPUTATION. Specifies adjacent imputation method.
    time_reference_col : str, optional
        Required only for time-based interpolation. Column should be in datetime format.
    verbose : bool, optional
        If True, prints details about the imputation process.
    """
    def __init__(self, 
                handle_missing_method : HandleMissingMethod = HandleMissingMethod.DROP,
                numeric_datatype_imputation_method : NumericDatatypeImputationMethod = 0,
                adjacent_imputation_method : AdjacentImputationMethod = 0 , 
                time_reference_col : str = "",
                verbose : bool = False
                ):
        
        self.handle_missing_method = handle_missing_method

        if handle_missing_method == HandleMissingMethod.DATATYPE_IMPUTATION and not numeric_datatype_imputation_method:
            raise ValueError("Numeric datatype imputation method must be selected in case of using DATATYPE_IMPUTATION as handling missing value.")
        if handle_missing_method == HandleMissingMethod.ADJACENT_VALUE_IMPUTATION and not adjacent_imputation_method:
            raise ValueError("Adjacent imputation method must be selected in case of using ADJACENT_VALUE_IMPUTATION as handling missing value.")
        
        self.numeric_datatype_imputation_method = numeric_datatype_imputation_method
        self.adjacent_imputation_method = adjacent_imputation_method
        self.time_reference_col = time_reference_col
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        match self.handle_missing_method:
            case HandleMissingMethod.DROP:
                return handle_missing_values_drop(data, self.verbose)
            case HandleMissingMethod.DATATYPE_IMPUTATION:
                return handle_missing_values_datatype_imputation(data, self.numeric_datatype_imputation_method, self.verbose)
            case HandleMissingMethod.ADJACENT_VALUE_IMPUTATION:
                return handle_missing_values_adjacent_value_imputation(data, self.adjacent_imputation_method, self.time_reference_col, self.verbose)
            case _:
                raise ValueError("Handling missing method is not valid. It should be selected using HandleMissingMethod enum.")