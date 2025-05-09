"""
Missing Values Handling Module
==============================

This module provides functionality for detecting and handling missing values in pandas DataFrames 
through configurable strategies, tailored for integration into preprocessing pipelines. It supports 
basic and advanced imputation methods, offering flexibility and robustness in managing incomplete data.

Core Features:
--------------
1. **Missing Value Dropping**
   - Removes rows containing any missing values.
   - Suitable for strict data quality requirements or small-scale datasets.

2. **Datatype-Based Imputation**
   - Imputes numeric columns using mean, median, or mode (as specified).
   - Imputes categorical columns using mode.
   - Ideal for general-purpose preprocessing with minimal domain assumptions.

3. **Adjacent Value Imputation**
   - Fills missing values using temporal or ordered context:
     - Forward fill
     - Backward fill
     - Linear interpolation
     - Time-based interpolation (requires datetime column)
   - Best for time series or sequence-based data.

4. **Pipeline Integration**
   - Includes `HandleMissingStep`, a `_PipelineStep`-compatible class for plug-and-play use in modular pipelines.

Enums:
------
- `HandleMissingMethod`: Strategy enum — DROP, DATATYPE_IMPUTATION, ADJACENT_VALUE_IMPUTATION.
- `NumericDatatypeImputationMethod`: Specifies numeric column strategy — MEAN, MEDIAN, MODE.
- `AdjacentImputationMethod`: Neighbor-based methods — FORWARD, BACKWARD, INTERPOLATION_LINEAR, INTERPOLATION_TIME.

Functions:
----------
- `handle_missing_values_drop`: Drops rows containing missing values.
- `handle_missing_values_datatype_imputation`: Fills missing values using datatype-aware rules.
- `handle_missing_values_adjacent_value_imputation`: Fills using adjacent/temporal values or interpolation.

Classes:
--------
- `HandleMissingStep`: Pipeline-compatible class that encapsulates the full missing value handling logic.
"""
import pandas as pd
from enum import Enum
from ..pipeline.pipeline import _PipelineStep


class AdjacentImputationMethod(Enum):
    """
    Enumeration of methods used to impute missing values based on adjacent data.

    These methods are typically applied to time series or ordered datasets
    where the value of a data point can be estimated from its neighbors.

    Attributes
    ----------
        FORWARD: int
            Forward fill — propagates the last valid (non-missing) observation forward.
        BACKWARD: int
            Backward fill — fills missing values using the next valid observation.
        INTERPOLATION_LINEAR: int
            Linear interpolation — estimates missing values by linearly interpolating between known values.
        INTERPOLATION_TIME: int
            Time-based interpolation — performs interpolation using a datetime index, suitable for time series data.

    Notes
    -----
    - Interpolation methods only apply to numeric columns.
    - Time-based interpolation requires a valid datetime column set as index.
    """
    # Enum classes make the code cleaner and avoid using invalid inputs
    FORWARD = 1
    BACKWARD = 2
    INTERPOLATION_LINEAR = 3
    INTERPOLATION_TIME = 4


class NumericDatatypeImputationMethod(Enum):
    """
    Enumeration of imputation methods for handling missing values in numeric columns.

    These methods are used to replace missing values with statistical summaries
    derived from the non-missing values in the column.

    Attributes
    ----------
        MEAN: int
            Impute using the mean (average) of the column.
        MEDIAN: int
            Impute using the median (middle value) of the column.
        MODE: int
            Impute using the mode (most frequent value) of the column.

    Notes
    -----
    - This enum is only applicable to numeric columns (e.g., int, float).
    - For categorical columns, the mode is used by default.
    - The `MODE` option will select the first mode when multiple modes exist.
    """
    # Enum classes make the code cleaner and avoid using invalid inputs
    # This enum specifies the method of imputation only for numeric columns, since for the categorical columns "MODE" is the only option
    MEAN = 1
    MEDIAN = 2
    MODE = 3


class HandleMissingMethod(Enum):
    """
    Enumeration of high-level strategies for handling missing values in a dataset.

    These strategies dictate how the pipeline should respond to the presence of missing data.

    Attributes
    ----------
        DROP: int
            Remove rows that contain any missing values.
        DATATYPE_IMPUTATION: int
            Impute missing values based on data types: numeric columns use a specified method (mean/median/mode), while categorical columns use the mode.
        ADJACENT_VALUE_IMPUTATION: int
            Impute missing values using neighboring data points via forward fill, backward fill, or interpolation.

    Notes
    -----
    - The `DATATYPE_IMPUTATION` strategy requires a `NumericDatatypeImputationMethod`.
    - The `ADJACENT_VALUE_IMPUTATION` strategy requires an `AdjacentImputationMethod`.
    - Time-based interpolation requires a valid datetime column.
    """
    DROP = 1
    DATATYPE_IMPUTATION = 2
    ADJACENT_VALUE_IMPUTATION = 3


def handle_missing_values_drop(data: pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    """
    Remove all rows from the DataFrame that contain any missing values.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing missing values.
    verbose : bool, optional
        If True, prints the number of rows before and after dropping, and a summary of missing values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with all rows containing missing values removed.

    Notes
    -----
    - This method is suitable when even a single missing value makes a record unusable.
    - Use with caution if data volume is limited.
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
    Impute missing values based on the data type of each column.

    - Numeric columns are filled using the specified statistical method (mean, median, or mode).
    - Non-numeric (categorical) columns are imputed using the mode (most frequent value).

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with potential missing values.
    numeric_datatype_imputation_method : NumericDatatypeImputationMethod
        Method to apply to numeric columns : MEAN, MEDIAN, or MODE
    verbose : bool, optional
        If True, prints missing value summaries before and after imputation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values imputed according to data types.

    Raises
    ------
    ValueError
        If an invalid imputation method is supplied for numeric columns.

    Notes
    -----
    - Mode selection for categorical columns uses the first mode if multiple modes exist.
    - This method supports both standard and mixed-type DataFrames.
    """
    result = data.copy()

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
    Impute missing values using neighboring (adjacent) values in the dataset.

    Supports techniques such as forward fill, backward fill, linear interpolation,
    and time-based interpolation (if a datetime column is provided).

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with missing values.
    adjacent_imputation_method : AdjacentImputationMethod
        Method used for imputation: FORWARD, BACKWARD, INTERPOLATION_LINEAR, or INTERPOLATION_TIME.
    time_reference_col : str, optional
        Required only for INTERPOLATION_TIME. Must be a valid datetime column.
    verbose : bool, optional
        If True, prints the number of missing values before and after imputation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values imputed using the specified adjacent value method.

    Raises
    ------
    ValueError
        If time-based interpolation is selected but no valid datetime column is provided.
    KeyError
        If the specified time reference column does not exist.

    Notes
    -----
    - Interpolation applies only to numeric columns.
    - Non-numeric columns are not affected by interpolation methods.
    - The time column is temporarily set as index for time-based interpolation and reset afterward.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Check for missing values
    # It is also possible to use isnull() instead of isna()

    result = data.copy()

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
    A modular pipeline step for handling missing values within a preprocessing pipeline.

    This class allows for configurable missing value handling and integrates seamlessly
    into data preprocessing workflows by implementing the `_PipelineStep` interface.

    Parameters
    ----------
    handle_missing_method : HandleMissingMethod
        The strategy used for handling missing values.
    numeric_datatype_imputation_method : NumericDatatypeImputationMethod, optional
        Required if `handle_missing_method` is DATATYPE_IMPUTATION. Specifies numeric column imputation method.
    adjacent_imputation_method : AdjacentImputationMethod, optional
        Required if `handle_missing_method` is ADJACENT_VALUE_IMPUTATION. Specifies the adjacent value method.
    time_reference_col : str, optional
        Required only for time-based interpolation. Should be a valid datetime column.
    verbose : bool, optional
        If True, prints detailed logs during execution.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame:
        Applies the configured missing value handling strategy to the input data.

    Raises
    ------
    ValueError
        If a required imputation method is not specified for the chosen strategy.

    Strategies supported
    ---------------------
    - Drop rows with missing values.
    - Impute using column data types (mean, median, or mode for numeric; mode for categorical).
    - Use adjacent values (forward/backward fill or interpolation).
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