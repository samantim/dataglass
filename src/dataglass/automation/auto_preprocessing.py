from ..preprocessing import *
import pandas as pd
from typing import List

def get_datatypes(data: pd.DataFrame) -> List:
    # Make a copy of data
    data_copy = data.copy()

    # Remove columns with 100% missing values
    for col in data_copy.columns:
        missing_ratio = calc_missing_ratio(data_copy, col)
        if missing_ratio == 1:
            data_copy = data_copy.drop(columns=[col])

    # Impute all NaN values with mode value to infer the column types correctly
    data_copy = handle_missing_values_datatype_imputation(data_copy, NumericDatatypeImputationMethod.MODE)

    # Datatype heuristic inference
    data_copy = convert_datatype_auto(data_copy)

    # Extract columns with specific types
    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    bool_columns = []

    data_types = data_copy.dtypes
    for col, dt in data_types.items():
        if pd.api.types.is_bool_dtype(dt):
            bool_columns.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dt):
            datetime_columns.append(col)
        elif pd.api.types.is_numeric_dtype(dt):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    # Find datetime dependent numeric columns based on their correlation with the datetime_reference columns
    correlation_columns = []
    correlation_columns.extend(numeric_columns)
    correlation_columns.extend(datetime_columns)
    corr = data_copy[correlation_columns].corr()
    # Check if it is a time series data
    if len(datetime_columns) > 0:
        time_reference_col = datetime_columns[0]
        # Correlation more than 0.3 is considered as dependency
        datetime_dependent_numeric_columns = [col for col in numeric_columns if corr.loc[time_reference_col, col] > 0.3]
    
    return data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns


def calc_missing_ratio(data: pd.DataFrame, col: str) -> float:
    # Calculate the ratio of missing values
    return data[col].isna().sum()/len(data[col])


def auto_handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # Get datatypes of all columns
    data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns = get_datatypes(data)

    # 1: Remove columns with 100% missing values
    for col in data.columns:
        missing_ratio = calc_missing_ratio(data, col)
        if missing_ratio == 1:
            data = data.drop(columns=[col])

    # 2: Handle missing values of datetime columns
    # Missing values are imputed by 1900-01-01 timestamp
    for col in datetime_columns:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(pd.Timestamp(year=1900, month=1, day=1))

    # 3: Datatype heuristic inference to unify all datetime formats
    data[datetime_columns] = convert_datatype_auto(data[datetime_columns])

    # 4: Handle categorical variables with more than 30% missing values
    for col in categorical_columns:
        missing_ratio = calc_missing_ratio(data, col)
        if missing_ratio >= 0.3:
            data[col] = data[col].fillna("Empty")
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    # 5: Impute bool values with False, as it is the safer value
    for col in bool_columns:
        data[col] = data[col].fillna(0)

    # 6: Handle numeric columns
    # Check if it is a time series data
    if len(datetime_columns) > 0:
        # If it is a time serie dataset
        # This scenario is only applied to datetime dependent numeric columns
        for col in datetime_dependent_numeric_columns:
            missing_ratio = calc_missing_ratio(data, col)
            if missing_ratio <= 0.1:
                # Interpolation by time whould be smooth
                data[col] = handle_missing_values_adjacent_value_imputation(data[[datetime_columns[0], col]], AdjacentImputationMethod.INTERPOLATION_TIME, datetime_columns[0])[col]
            else:
                # Imputation by ffill and bfill is more reasonable (bfill is used to ensure that the first row is also imputed)
                data[col] = handle_missing_values_adjacent_value_imputation(data[[col]], AdjacentImputationMethod.FORWARD)
                data[col] = handle_missing_values_adjacent_value_imputation(data[[col]], AdjacentImputationMethod.BACKWARD)

    # If it is not a TS dataset or 
    # there are some datetime independent columns or 
    # there are some NaN values remained from the TS related approaches
    for col in numeric_columns:
        skew = abs(data[col].skew())
        # Skew < 1 is considered as partially normally distributed, and mean can be valid for missing values
        if skew < 1:
            data[col] = handle_missing_values_datatype_imputation(data[[col]], NumericDatatypeImputationMethod.MEAN)
        else:
            # In skewed distributions, median is a better choice for imputation
            data[col] = handle_missing_values_datatype_imputation(data[[col]], NumericDatatypeImputationMethod.MODE)

    # 7: This datatype change is necessary to reset datatypes to their own nature, because they may be changed unintentionally in interpolation
    for col in data.columns:
        data[col] = data[col].astype(data_types[col])

    return data
    



