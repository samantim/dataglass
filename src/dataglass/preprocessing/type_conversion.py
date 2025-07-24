"""
Type Conversion Module
=======================

This module provides functionality to convert column datatypes in pandas DataFrames, 
either automatically or based on user-defined scenarios. It is designed to be integrated 
into modular data preprocessing pipelines and ensures that datatypes are appropriate 
for modeling, analysis, and feature engineering.

Core Features:
--------------
1. **Automatic Datatype Conversion**
   - Automatically converts object-type columns to numeric (int or float) if possible.
   - If numeric conversion fails, attempts to convert to datetime.
   - Leaves non-convertible columns unchanged.

2. **User-Defined Datatype Conversion**
   - Enables explicit datatype conversion for selected columns.
   - Supports conversion to int, float, or datetime (with custom format strings).
   - Includes extensive error handling and validation.

3. **Pipeline Integration**
   - Includes `TypeConversionStep`, which implements `_PipelineStep` for integration into 
     end-to-end data pipelines.

Enums:
------
- `ConvertDatatypeMethod`: Strategy enum — AUTO or USER_DEFINED.

Functions:
----------
- `convert_datatype_auto`: Automatically infers and converts object-type columns.
- `convert_datatype_userdefined`: Converts columns to target datatypes based on a user-specified scenario.

Classes:
--------
- `TypeConversionStep`: Pipeline-compatible class for orchestrating datatype conversion.
"""
import pandas as pd
from typing import Dict
from enum import Enum
from ..pipeline.pipeline import _PipelineStep
import warnings

class ConvertDatatypeMethod(Enum):
    """
    Enum for specifying the strategy used to convert data types.

    Attributes
    ----------
        AUTO: int
            Automatically infers and converts data types based on content heuristics.
        USER_DEFINED: int
            Applies datatype conversions based on an explicit user-provided specification.
    """
    AUTO = 1
    USER_DEFINED = 2


def convert_datatype_auto(data : pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    """
    Automatically attempts to convert object-type columns to numeric (int/float) or datetime types.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame to process.
        verbose (bool, optional): If True, prints datatypes before and after conversion.

    Returns
    ----------
        pd.DataFrame: A new DataFrame with updated datatypes where applicable.

    Notes
    ----------
    - Invalid conversions are silently skipped.
    - Integer casting only occurs when it is safe (i.e., all values are whole numbers).
    
    Conversion Logic
    ----------
    - If a column is of object dtype, try converting to numeric.
    - If the column is numeric with no fractional values, downcast float to int.
    - If numeric conversion fails, attempt to convert to datetime.
    - Columns that cannot be converted remain unchanged.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Show the data types before applying any conversion

    result = data.copy()

    if verbose:
        print(f"Before automatic datatype conversion, the datatype are as follows:\n{result.dtypes}")

    # If the data type of the columns is not numeric, it try to convert it to datatime type
    # If the column content is not datetime no changes will happen
    for col in result.columns:
        try:
            # Convert data type of the numeric-like columns which has object type
            if result[col].dtype == "object":
                result[col] = pd.to_numeric(result[col])
            # If the type is float but it is not necessary based on the column contents, it will cast to int
            if pd.api.types.is_float_dtype(result[col]):
                # Check if there is any fractional part in the column contents, if there is not any, it is safe to convert to int (otherwise it leads to data loss)
                if (result[col].dropna() % 1 == 0).all():
                    result[col] = result[col].astype("int64")
        except:
            pass

        try:
            # Ignore warnings
            with warnings.catch_warnings(action="ignore"):
                if not pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = pd.to_datetime(result[col], format="mixed", dayfirst=True)
        except:
            pass

    # Show the data types after applying auto conversions
    if verbose:
        print(f"After automatic datatype conversion, the datatype are as follows:\n{result.dtypes}")

    return result    


def convert_datatype_userdefined(data : pd.DataFrame, convert_scenario : Dict, verbose : bool = False) -> pd.DataFrame:
    """
    Converts specified columns in a DataFrame to target datatypes using a user-defined scenario.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame.
        convert_scenario (Dict): A dict defining conversion rules. Example:
        {
            "column": ["Age", "Start Date"],
            "datatype": ["int", "datetime"],
            "format": ["", "%Y-%m-%d"]
        }
        verbose (bool, optional): If True, prints datatypes before and after conversion.

    Returns
    ----------
        pd.DataFrame
            A DataFrame with updated column datatypes as defined.

    Raises
    ----------
    ValueError:
    - If any listed column does not exist.
    - If unsupported datatypes are specified.
    - If datetime format is missing or incorrectly applied.
    - If any actual conversion fails due to data inconsistency.

    Notes
    ----------
    - Leading/trailing whitespaces in the scenario values are stripped.
    - Only "int", "float", and "datetime" datatypes are supported.
    - Format strings are required for datetime and must follow ISO 8601 standards.
    """
    # Sample conver_scenario:
    # {"column":["High School Percentage", "Test Date"],
    #  "datatype":["int", "datetime"],
    #  "format":["", "%m/%d/%Y"] }

    result = data.copy()
    
    # Strip whitespaces
    for key in convert_scenario.keys():
        convert_scenario[key] = [item.strip() for item in convert_scenario[key]]

    # Show the data types before applying any conversion
    if verbose:
        print(f"Before automatic datatype conversion, the datatype are as follows:\n{result.dtypes}")
    
    # Check all the provided column names to be in the dataset
    if not all(col in result.columns for col in convert_scenario["column"]):
        raise ValueError("At least one of the columns provided in the scenario is not valid!")
    
    # Check all the provided datatypes to be valid
    if not all(dt in ["float", "int", "datetime"] for dt in convert_scenario["datatype"]):
        raise ValueError("At least one of the type provided in the scenario is not valid! The only acceptable data types are: {float, int, datetime}")
    
    # create a list of tuples based on the convert_scenario dict
    # zip command creates tuple of the elements of a row
    # conversion to a list is necessary since zipped output can be consumed only once, but we need it more
    convert_scenario_zipped = list(zip(convert_scenario["column"], convert_scenario["datatype"], convert_scenario["format"]))

    # Check if there is a non datetime column which has provided format (we accept format only for datetime conversion)
    if any(ft != "" and dt != "datetime" for _ , dt, ft in convert_scenario_zipped):
        raise ValueError("Only datetime conversion accepts format (ISO 8601)")
    
    # Check all the datetime conversions have format
    if any(ft == "" and dt == "datetime" for _ , dt, ft in convert_scenario_zipped):
        raise ValueError("Datetime conversion needs format (ISO 8601)")
    
    for col, dt, ft in convert_scenario_zipped:
        try:
            match dt:
                case "int":
                    # If the current type is not int, the conversion will apply
                    if not pd.api.types.is_integer_dtype(result[col]):
                        result[col] = result[col].astype("int")
                case "float":
                    # If the current type is not int, the conversion will apply
                    if not pd.api.types.is_float_dtype(result[col]):
                        result[col] = result[col].astype("float")
                case "datetime":
                    # If the current type is not numeric and datetime, the conversion will apply
                    if not pd.api.types.is_datetime64_any_dtype(result[col]) and not pd.api.types.is_numeric_dtype(result[col]):
                        result[col] = pd.to_datetime(result[col], format=ft)
        except Exception as e:
            raise ValueError(f"Conversion failed for column '{col}' with error: {e}")
    
    # Show the data types after applying auto conversions
    if verbose:
        print(f"After automatic datatype conversion, the datatype are as follows:\n{result.dtypes}")

    return result


class TypeConversionStep(_PipelineStep):
    """
    A pipeline-compatible step for column datatype conversion.

    Facilitates datatype standardization in a preprocessing pipeline. Supports both
    automatic and user-defined conversion strategies. Integrates seamlessly into
    modular data pipelines for feature engineering and modeling.

    Parameters
    ----------
    convert_datatype_method : ConvertDatatypeMethod
        Strategy for conversion: AUTO (automatic inference) or USER_DEFINED (explicit mapping).
    
    convert_scenario : Dict, optional
        Required if using USER_DEFINED method. Should include:
            - "column": list of column names
            - "datatype": list of target datatypes ("int", "float", "datetime")
            - "format": list of datetime formats ("" for non-datetime columns)

    verbose : bool, optional
        If True, prints datatypes before and after conversion.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame:
        Executes the datatype conversion on the provided DataFrame.

    Raises
    ------
        ValueError: 
            If parameters are missing or invalid in USER_DEFINED mode.
    """
    def __init__(self, 
                convert_datatype_method : ConvertDatatypeMethod,
                convert_scenario : Dict = None,
                verbose : bool = False
                ):
        
        self.convert_datatype_method = convert_datatype_method
        self.convert_scenario = convert_scenario
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        match self.convert_datatype_method:
            case ConvertDatatypeMethod.AUTO:
                return convert_datatype_auto(data, self.verbose)
            case ConvertDatatypeMethod.USER_DEFINED:
                if not self.convert_scenario:
                    raise ValueError("Converting scenario is mandatory in user-defined mode.")
                return convert_datatype_userdefined(data, self.convert_scenario, self.verbose)