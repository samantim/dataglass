"""
Type Conversion Module
=======================

This module provides functionality for converting column datatypes in pandas DataFrames 
as part of a modular data preprocessing pipeline. It includes automatic and user-defined 
conversion strategies to ensure that data types are appropriate for downstream analysis 
and modeling tasks.

Core Features:
--------------
1. **Automatic Datatype Conversion**
   - Attempts to convert object-type columns to numeric. If that fails, tries datetime conversion.
   - Columns that cannot be converted remain unchanged.

2. **User-Defined Datatype Conversion**
   - Allows users to define a specific conversion scenario including column names, target data types 
     (int, float, datetime), and optional datetime formats.
   - Ensures accurate type transformation with validation and error handling.

3. **Pipeline Integration**
   - Includes the `TypeConversionStep` class, which implements the `PipelineStep` interface 
     for seamless integration into preprocessing pipelines.

Enums:
------
- `ConvertDatatypeMethod`: Defines strategies for datatype conversion (AUTO or USER_DEFINED).

Functions:
----------
- `convert_datatype_auto`: Converts object-type columns to numeric or datetime using automatic inference.
- `convert_datatype_userdefined`: Converts specified columns to target datatypes based on a provided scenario dictionary.

Classes:
--------
- `TypeConversionStep`: A class implementing the `PipelineStep` interface for converting column datatypes 
  within a data pipeline.
"""
import pandas as pd
from typing import Dict
from enum import Enum
from ..pipeline.pipeline import _PipelineStep
import warnings

class ConvertDatatypeMethod(Enum):
    """
    Enum for specifying the method to convert data types.

    Attributes:
        AUTO (int): Automatically infers and converts data types based on the input data.
        USER_DEFINED (int): Converts data types based on user-provided convert scenario.
    """
    AUTO = 1
    USER_DEFINED = 2


def convert_datatype_auto(data : pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    """
    Automatically attempts to convert object-type columns to numeric or datetime types.

    It first tries to convert object-type columns to numeric. If that fails, it attempts to convert 
    them to datetime. Columns that cannot be converted remain unchanged.

    Args:
        data (pd.DataFrame): Input DataFrame to process.
        verbose (bool, optional): If True, prints datatypes before and after conversion. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with updated datatypes where conversions were successful.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Show the data types before applying any conversion
    if verbose:
        print(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    # If the data type of the columns is not numeric, it try to convert it to datatime type
    # If the column content is not datetime no changes will happen
    for col in data.columns:
        try:
            # Convert data type of the numeric-like columns which has object type
            if data[col].dtype == "object":
                data[col] = pd.to_numeric(data[col])
        except:
            pass

        try:
            with warnings.catch_warnings(action="ignore"):
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_datetime(data[col])
        except:
            pass

    # Show the data types after applying auto conversions
    if verbose:
        print(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    return data    


def convert_datatype_userdefined(data : pd.DataFrame, convert_scenario : Dict, verbose : bool = False) -> pd.DataFrame:
    """
    Converts specific columns in a DataFrame to user-defined data types based on a provided scenario.

    The scenario dictionary must include the column names, target data types ("int", "float", "datetime"),
    and an optional format string (required for datetime conversion).

    Args:
        data (pd.DataFrame): Input DataFrame to convert.
        convert_scenario (Dict): Dictionary with keys "column", "datatype", and "format". Example:
            {
                "column": ["Score", "Date"],
                "datatype": ["float", "datetime"],
                "format": ["", "%Y-%m-%d"]
            }
        verbose (bool, optional): If True, prints datatypes before and after conversion. Defaults to False.

    Raises:
        ValueError: If a column is missing, type is unsupported, or format is misused.
        ValueError: If any conversion fails.

    Returns:
        pd.DataFrame: DataFrame with columns converted according to the provided scenario.
    """
    # Sample conver_scenario:
    # {"column":["High School Percentage", "Test Date"],
    #  "datatype":["int", "datetime"],
    #  "format":["", "%m/%d/%Y"] }

    # Strip whitespaces
    for key in convert_scenario.keys():
        convert_scenario[key] = [item.strip() for item in convert_scenario[key]]

    # Show the data types before applying any conversion
    if verbose:
        print(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")
    
    # Check all the provided column names to be in the dataset
    if not all(col in data.columns for col in convert_scenario["column"]):
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
                    if not pd.api.types.is_integer_dtype(data[col]):
                        data[col] = data[col].astype("int")
                case "float":
                    # If the current type is not int, the conversion will apply
                    if not pd.api.types.is_float_dtype(data[col]):
                        data[col] = data[col].astype("float")
                case "datetime":
                    # If the current type is not numeric and datetime, the conversion will apply
                    if not pd.api.types.is_datetime64_any_dtype(data[col]) and not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col], format=ft)
        except Exception as e:
            raise ValueError(f"Conversion failed for column '{col}' with error: {e}")
    
    # Show the data types after applying auto conversions
    if verbose:
        print(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    return data


class TypeConversionStep(_PipelineStep):
    """
    Pipeline step for converting column datatypes using either automatic inference 
    or user-defined rules.

    This class integrates with the DataPipeline system and allows users to standardize
    column types, which is critical for analysis and modeling. It supports:
    - Automatic conversion of object-type columns to numeric or datetime.
    - Manual conversion using user-defined column, type, and format specifications.

    Parameters
    ----------
    convert_datatype_method : ConvertDatatypeMethod
        Strategy for converting datatypes. Options: AUTO or USER_DEFINED.
    convert_scenario : Dict
        Required if `convert_datatype_method` is USER_DEFINED. Should include:
            - "column": list of column names,
            - "datatype": list of target datatypes ("int", "float", "datetime"),
            - "format": list of datetime formats (use empty string for non-datetime).
    verbose : bool, optional
        If True, prints datatype information before and after conversion.
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