import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import Dict


def convert_datatype_auto(data : pd.DataFrame) -> pd.DataFrame:
    # Show the data types before applying any conversion
    logging.info(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

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
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_datetime(data[col])
        except:
            pass

    # Show the data types after applying auto conversions
    logging.info(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    return data    

def convert_datatype_ud(data : pd.DataFrame, convert_scenario : Dict) -> pd.DataFrame:
    # Sample conver_scenario:
    # {"column":["High School Percentage", "Test Date"],
    #  "datatype":["int", "datetime"],
    #  "format":["", "%m/%d/%Y"] }

    # Strip whitespaces
    for key in convert_scenario.keys():
        convert_scenario[key] = [item.strip() for item in convert_scenario[key]]

    # Show the data types before applying any conversion
    logging.info(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")
    
    # Check all the provided column names to be in the dataset
    if not all(col in data.columns for col in convert_scenario["column"]):
        logging.error("At least one of the columns provided in the scenario is not valid!")
        return data
    
    # Check all the provided datatypes to be valid
    if not all(dt in ["float", "int", "datetime"] for dt in convert_scenario["datatype"]):
        logging.error("At least one of the type provided in the scenario is not valid! The only acceptable data types are: {float, int, datetime}")
        return data
    
    # create a list of tuples based on the convert_scenario dict
    # zip command creates tuple of the elements of a row
    # conversion to a list is necessary since zipped output can be consumed only once, but we need it more
    convert_scenario_zipped = list(zip(convert_scenario["column"], convert_scenario["datatype"], convert_scenario["format"]))

    # Check if there is a non datetime column which has provided format (we accept format only for datetime conversion)
    if any(ft != "" and dt != "datetime" for _ , dt, ft in convert_scenario_zipped):
        logging.error("Only datetime conversion accepts format (ISO 8601)")
        return data
    
    # Check all the datetime conversions have format
    if any(ft == "" and dt == "datetime" for _ , dt, ft in convert_scenario_zipped):
        logging.error("Datetime conversion needs format (ISO 8601)")
        return data
    
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
            logging.error(f"Conversion failed for column '{col}' with error: {e}")
            return data
    
    # Show the data types after applying auto conversions
    logging.info(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    return data