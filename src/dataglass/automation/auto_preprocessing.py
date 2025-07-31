from ..preprocessing import *
import pandas as pd
from typing import List, Dict
import logging

pd.set_option('future.no_silent_downcasting', True)

# Config logging 
logging.basicConfig(filename="auto_preprocess.log", filemode="w", level=logging.INFO, format="%(message)s")

def _get_datatypes(data: pd.DataFrame, verbose: bool = False) -> List:
    # It returns all column datatypes as well as seperating them in different list based on their types

    # Find columns with 100% missing values
    empty_columns = []
    for col in data.columns:
        missing_ratio = _calc_missing_ratio(data, col)
        if missing_ratio == 1:
            empty_columns.append(col)
    # Remove empty columns
    data_copy = data.drop(columns=empty_columns)

    # Impute all NaN values with mode value to infer the column types correctly
    data_copy = handle_missing_values_datatype_imputation(data_copy, NumericDatatypeImputationMethod.MODE)

    # Datatype heuristic inference
    data_copy = convert_datatype_auto(data_copy)

    # Extract columns with specific types
    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    bool_columns = []
    datetime_dependent_numeric_columns = []

    # Create datatype separate lists 
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
        datetime_dependent_numeric_columns = [col for col in numeric_columns if abs(corr.loc[time_reference_col, col]) > 0.3]
    
    # Log data shape and types before any change
    if verbose: logging.info(f"Data shape is: {data.shape}\n\nInitial datatypes are:\n{data.dtypes}")

    return data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns


def _calc_missing_ratio(data: pd.DataFrame, col: str) -> float:
    # Calculate the ratio of missing values of a column
    return data[col].isna().sum()/len(data[col])


def _upcast_int_columns(data: pd.DataFrame, numeric_columns: List) -> pd.DataFrame:
    # Upcast numeric int columns to float to avoid imputation errors
    for col in numeric_columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = data[col].astype(float)
    
    return data


def _reset_datatypes_to_default(data: pd.DataFrame, data_types: List) -> pd.DataFrame:
    # This datatype change is necessary to reset datatypes to their own nature, because they may be changed unintentionally in interpolation
    for col in data.columns:
        data[col] = data[col].astype(data_types[col])
    
    return data


def auto_handle_missing_values(data: pd.DataFrame, data_types: List, numeric_columns: List, categorical_columns: List,
                               datetime_columns: List, bool_columns: List, datetime_dependent_numeric_columns: List, verbose: bool = False) -> pd.DataFrame:
    # It handles all missing values automatically in different steps

    if verbose: logging.info("\nHandling missing values:")

    # 0: Upcast numeric int columns to float to avoid imputation errors
    data = _upcast_int_columns(data, numeric_columns)

    # 1: Remove columns with 100% missing values
    for col in data.columns:
        missing_ratio = _calc_missing_ratio(data, col)
        if missing_ratio == 1:
            data = data.drop(columns=[col])
            if verbose: logging.info(f"Column '{col}' was empty and is deleted.")

    # 2: Handle missing values of datetime columns
    # Missing values are imputed by 1900-01-01 timestamp
    for col in datetime_columns:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(pd.Timestamp(year=1900, month=1, day=1))
            if verbose: logging.info(f"Column '{col}' had some NaN values which are imputed by default value '01-01-1900'.")

    # 3: Datatype heuristic inference to unify all datetime formats
    data[datetime_columns] = convert_datatype_auto(data[datetime_columns])

    # 4: Handle categorical variables with more than 30% missing values
    for col in categorical_columns:
        missing_ratio = _calc_missing_ratio(data, col)
        if missing_ratio >= 0.3:
            data[col] = data[col].fillna("Empty")
            if verbose: logging.info(f"Column '{col}' is categorical and had more than 30% NaN values which are imputed by default value 'Empty'.")
        elif missing_ratio > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
            if verbose: logging.info(f"Column '{col}' is categorical and had less than 30% NaN values which are imputed by the most used value of the column (mode).")


    # 5: Impute bool values with False, as it is the safer value
    for col in bool_columns:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(0)
            if verbose: logging.info(f"Column '{col}' is boolean and missing values are imputed by default value 'False'.")


    # 6: Handle numeric columns
    # Check if it is a time series data
    if len(datetime_columns) > 0:
        # If it is a time serie dataset
        # This scenario is only applied to datetime dependent numeric columns
        for col in datetime_dependent_numeric_columns:
            missing_ratio = _calc_missing_ratio(data, col)
            if missing_ratio > 0:
                if missing_ratio <= 0.1:
                    # Interpolation by time whould be smooth
                    data[col] = handle_missing_values_adjacent_value_imputation(data[[datetime_columns[0], col]], AdjacentImputationMethod.INTERPOLATION_TIME, datetime_columns[0])[col]
                    if verbose: logging.info(f"Column '{col}' is numeric and missing values are imputed by time-based interpolation. Because the dataset is a time-series and '{col}' is datetime-dependent.")
                else:
                    # Imputation by ffill and bfill is more reasonable (bfill is used to ensure that the first row is also imputed)
                    data[col] = handle_missing_values_adjacent_value_imputation(data[[col]], AdjacentImputationMethod.FORWARD)
                    data[col] = handle_missing_values_adjacent_value_imputation(data[[col]], AdjacentImputationMethod.BACKWARD)
                    if verbose: logging.info(f"Column '{col}' is numeric and missing values are imputed by forward/backward fill, due to large number of missings, although the dataset is a time-series and '{col}' is datetime-dependent.")


    # If it is not a TS dataset or 
    # there are some datetime independent columns or 
    # there are some NaN values remained from the TS related approaches
    for col in numeric_columns:
        if data[col].isna().sum() > 0:
            skew = abs(data[col].skew())
            # Skew < 1 is considered as partially normally distributed, and mean can be valid for missing values
            if skew < 1:
                data[col] = handle_missing_values_datatype_imputation(data[[col]], NumericDatatypeImputationMethod.MEAN)
                if verbose: logging.info(f"Column '{col}' is numeric and missing values are imputed by mean, as the distribution is not very skewed.")
            else:
                # In skewed distributions, median is a better choice for imputation
                data[col] = handle_missing_values_datatype_imputation(data[[col]], NumericDatatypeImputationMethod.MEDIAN)
                if verbose: logging.info(f"Column '{col}' is numeric and missing values are imputed by median, as the distribution is strongly skewed.")


    # 7: This datatype change is necessary to reset datatypes to their own nature, because they may be changed unintentionally in interpolation
    data = _reset_datatypes_to_default(data, data_types)

    return data


def auto_handle_duplicates(data: pd.DataFrame, categorical_columns: List, datetime_columns: List, verbose: bool = False) -> pd.DataFrame:
    # It eliminates duplicate values automatically based on exact and fuzzy matching

    if verbose: logging.info("\nHandling duplicates:")

    n_rows = data.shape[0]

    # 1: Apply exact duplicate removal based on all columns
    data = handle_duplicate_values_exact(data)
    if verbose and n_rows - data.shape[0] > 0: logging.info(f"{n_rows - data.shape[0]} row(s) are dropped as duplicates, due to exact matching.")

    n_rows = data.shape[0]

    # 2: Apply fuzzy matching for categorical and datetime columns
    fuzzymatching_columns = []
    fuzzymatching_columns.extend(categorical_columns)
    fuzzymatching_columns.extend(datetime_columns)

    data = handle_duplicate_values_fuzzy(data, fuzzymatching_columns, (95,100))
    if verbose and n_rows - data.shape[0] > 0: logging.info(f"{n_rows - data.shape[0]} row(s) are dropped as duplicates, due to fuzzy matching with at least 95% similarity.")

    return data


def _decide_outlier_detection_method(data: pd.DataFrame, numeric_columns: List, verbose: bool = False) -> Dict[str, HandleOutlierMethod | str]:
    # It automatically decides outlier detection method for each numeric column (univariate)
    # or for the whole set (multivariate), based on dataset characteristics.
    # Assumes missing values are already handled.
    
    methods = {}
    n_samples = data.shape[0]
    n_features = len(numeric_columns)

    # Decide between per-column or multivariate detection
    force_per_column = (
        n_features == 1 or # Only one feature presents
        n_samples < 200 or # There is only less than 200 samples (low number of samples)
        any(data[col].nunique() < 10 for col in numeric_columns) # At least one of numeric columns has less than 10 unique values
    )

    if force_per_column:
        # Per-column (univariate) strategy: only IQR and Z-score are appropriate
        for col in numeric_columns:
            skew = data[col].skew()
            n_unique = data[col].nunique()

            if (abs(skew) <= 1 and # Not very skewed
                n_samples >= 100 and # Have at least 100 recods
                n_unique >= 20 # Have more than 20 unique values
            ):
                methods[col] = DetectOutlierMethod.ZSCORE
            else:
                methods[col] = DetectOutlierMethod.IQR
    else:
        # Multivariate strategy: LOF or Isolation Forest
        high_dimensional = n_features > 20
        large_dataset = n_samples > 10000

        if high_dimensional or large_dataset:
            methods["__multivariate__"] = DetectOutlierMethod.ISOLATION_FOREST
        else:
            methods["__multivariate__"] = DetectOutlierMethod.LOCAL_OUTLIER_FACTOR

    return methods


def _decide_outlier_handling_method(data: pd.DataFrame, detected_outliers: Dict[str, List[int]]) -> Dict[str, HandleOutlierMethod | None]:
    # Decide whether and how to handle outliers per column using only data characteristics.

    methods = {}

    for col, idxs in detected_outliers.items():

        # Check data characteristics
        outliers_ratio = len(idxs) / len(data[col])
        skew = data[col].skew()
        std = data[col].std()
        iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
        rel_range = (data[col].max() - data[col].min()) / (data[col].median() + 1e-6)

        if (
            outliers_ratio < 0.05 and # very few points
            rel_range > 10 and # huge spread vs. centre
            abs(skew) > 2 and # very strongly skewed
            std > iqr * 2 # heavy tails
        ):
            methods[col] = None # No need to action because the outliers are likely meaningful variations (e.g., Income, Product sales volume, and Web page views)
        elif abs(skew) > 1:
            methods[col] = HandleOutlierMethod.REPLACE_WITH_MEDIAN # strongly skewed
        else:
            methods[col] = HandleOutlierMethod.CAP_WITH_BOUNDARIES # otherwise keep shape, just cap

    return methods


def auto_handle_outliers(data: pd.DataFrame, numeric_columns: List, data_types: List, verbose: bool = False) -> pd.DataFrame:
    # Perform detection and handling of outliers based on detection and handling methods decided

    if verbose: logging.info("\nHandling outliers:")

    # Get the detection methods for each columns or multivariate
    detection_methods = _decide_outlier_detection_method(data, numeric_columns, verbose=True)

    # Discover if it's multivariate or per-column
    is_multivariate = "__multivariate__" in detection_methods

    if is_multivariate:
        # Multivariate outlier detection
        detection_method = detection_methods["__multivariate__"]
        # Detect outlier indecies based on all numeric columns of dataset
        outliers, _ = detect_outliers(data, detect_outlier_method=detection_method, columns_subset=numeric_columns)

        # Drop all outlier rows as they are meaningful together
        if bool(outliers):
            n_rows = data.shape[0]
            data = data.drop(index=outliers[numeric_columns[0]])
            if verbose and n_rows - data.shape[0] > 0: logging.info(f"{n_rows - data.shape[0]} row(s) are deleted as outliers based on detection method '{detection_method.name}'.")

    else:
        # Per-column univariate detection

        # Upcast numeric int columns to float to avoid imputation errors
        data = _upcast_int_columns(data, numeric_columns)

        for col, detection_method in detection_methods.items():
            # Detect outlier indecies and inlier boundaries for each numeric column of dataset
            outliers, boundaries = detect_outliers(data[[col]], detect_outlier_method=detection_method)
            if col in outliers.keys():
                # Decide how to handle each column
                handling_method = _decide_outlier_handling_method(data[[col]], outliers)[col]

                if not handling_method is None: # If there is need for actions
                    data[col] = handle_outliers(data=data[[col]], outliers=outliers, boundaries=boundaries, handle_outlier_method=handling_method)
                    if verbose: logging.info(f"Outliers of the column '{col}' are found based on detection method '{detection_method.name}' and handled by '{handling_method.name}' method.")
                else:
                    if verbose: logging.info(f"Although column '{col}' has some outliers based on detection method '{detection_method.name}', there is no need to action.")

        
        # This datatype change is necessary to reset datatypes to their own nature, because they may be changed unintentionally in interpolation
        data = _reset_datatypes_to_default(data, data_types)

    return data


def auto_preprocess_for_analysis(data: pd.DataFrame, verbose: bool=False):
    input_data = data.copy()

    # Get datatypes of all columns
    data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns = _get_datatypes(input_data, verbose)

    # Handle missing values
    result = auto_handle_missing_values(input_data, data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns, verbose)
    # Handle duplicates
    result = auto_handle_duplicates(result, categorical_columns, datetime_columns, verbose)
    # Handle outliers
    result = auto_handle_outliers(result, numeric_columns, data_types, verbose)
    # Infer datatype conversions
    result = convert_datatype_auto(result)

    # Log data shape and types after preprocessing
    if verbose: logging.info(f"\nData shape after preprocessing is: {result.shape}\n\nNew datatypes are:\n{result.dtypes}")

    return result


