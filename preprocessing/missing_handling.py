import pandas as pd
from enum import Enum


class AdjacentImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    FORWARD = 1
    BACKWARD = 2
    INTERPOLATION_LINEAR = 3
    INTERPOLATION_TIME = 4


class NumericDatatypeImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    # This enum specifies the method of imputation only for numeric columns, since for the categorical columns "MODE" is the only option
    MEAN = 1
    MEDIAN = 2
    MODE = 3


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
    
    # Display dataset info before and after imputation if verbose is enabled
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    for col in data.columns:
        # Check if the columns is numeric
        if pd.api.types.is_numeric_dtype(data[col]):
            match numeric_datatype_imputation_method:
                case NumericDatatypeImputationMethod.MEAN:
                    data[col] = data[col].fillna(data[col].mean())
                case NumericDatatypeImputationMethod.MEDIAN:
                    data[col] = data[col].fillna(data[col].median())
                case NumericDatatypeImputationMethod.MODE:
                    # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
                    data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # If it is a categorical columns
            # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
            data[col] = data[col].fillna(data[col].mode()[0])

    # Check dataset after dropping missing values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


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
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Fill missing values using adjacent value imputation
    match adjacent_imputation_method:
        case AdjacentImputationMethod.FORWARD:
            # Fill missing values using forward fill (Note that fillna(method='ffill') method is deprecated)
            data = data.ffill()

        case AdjacentImputationMethod.BACKWARD:
            # Fill missing values using backward fill (Note that fillna(method='bfill') method is deprecated)
            data = data.bfill()

        case AdjacentImputationMethod.INTERPOLATION_LINEAR:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Fill missing values using linear interpolation (Default method is linear)
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='linear')

        case AdjacentImputationMethod.INTERPOLATION_TIME:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Check if time reference column is provided, as it is needed for time interpolation
            if not time_reference_col:
                raise ValueError("Time reference column is required for time interpolation.")
            else:
                try:
                    # Convert time reference column to datetime if contains datatime values, otherwise it will raise an error
                    data[time_reference_col] = pd.to_datetime(data[time_reference_col])
                except ValueError:
                    raise ValueError(
                        f"The column '{time_reference_col}' is not in a valid datetime format. "
                        f"This method requires a datetime column for time-based interpolation."
                    )
                                
            # Setting the datetime column as index is necessary for time-based interpolation
            data = data.set_index(time_reference_col)

            # Fill missing values using time interpolation
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='time')
            
            # Reset index to original
            data = data.reset_index()

    # Check dataset after dropping missing values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data