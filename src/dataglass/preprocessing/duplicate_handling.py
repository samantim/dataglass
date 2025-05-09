"""
Duplicate Handling Module
==========================

This module provides tools for detecting and removing duplicate rows in pandas DataFrames, 
designed for integration into modular data preprocessing pipelines. It supports both exact 
and fuzzy (approximate) duplicate detection based on customizable thresholds.

Core Features
-------------
1. **Exact Duplicate Removal**
   - Removes fully identical rows.
   - Optionally limits the check to specific columns.

2. **Fuzzy Duplicate Removal**
   - Identifies near-duplicate rows using string similarity scoring.
   - Allows configurable similarity thresholds and subset of columns to focus matching.
   - Retains the first detected entry and removes other similar rows.

3. **Pipeline Integration**
   - Includes `HandleDuplicateStep`, a class implementing the `PipelineStep` interface, 
     enabling easy inclusion in broader machine learning or data cleaning workflows.

Enums
-----
- `HandleDuplicateMethod` : Enum specifying duplicate handling methods (`EXACT` or `FUZZY`).

Functions
---------
- `handle_duplicate_values_exact(data, columns_subset=None, verbose=False)`
    Removes fully identical rows from a DataFrame.
- `handle_duplicate_values_fuzzy(data, columns_subset=None, similarity_thresholds=None, verbose=False)`
    Detects and removes near-duplicates based on average text similarity.

Classes
-------
- `HandleDuplicateStep`
    A configurable pipeline step for applying exact or fuzzy duplicate handling.

Notes
-----
- Fuzzy matching has quadratic time complexity (O(n²)); it can be slow on large datasets (>10,000 rows).
- Preprocessing such as text normalization or column selection can improve fuzzy matching accuracy.
"""
import pandas as pd
from typing import List, Tuple
from rapidfuzz import fuzz
from itertools import combinations
from enum import Enum
from ..pipeline.pipeline import _PipelineStep

class HandleDuplicateMethod(Enum):
    """
    Enumeration of methods to handle duplicate values in a dataset.

    Attributes
    ----------
    EXACT : int
        Identifies and removes exact duplicates.
    FUZZY : int
        Identifies and removes approximate (fuzzy) duplicates based on string similarity.
    """
    EXACT = 1
    FUZZY = 2

def handle_duplicate_values_exact(data : pd.DataFrame, columns_subset : List = None, verbose : bool = False) -> pd.DataFrame:
    """
    Remove exact duplicate rows from a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to clean.
    columns_subset : list of str, optional
        Columns to consider when checking for duplicates.
        If None, all columns are used.
    verbose : bool, default=False
        If True, displays the number of duplicates found, 
        a sample of duplicate rows, and the dataset size before and after removal.

    Returns
    -------
    pd.DataFrame
        A DataFrame with exact duplicates removed, keeping the first occurrence in each group.

    Raises
    ------
    ValueError
        If any specified columns in `columns_subset` do not exist in the input DataFrame.

    Notes
    -----
    - Only fully identical rows across the selected columns are treated as duplicates.
    - Leading/trailing whitespace in column names is automatically stripped.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Check dataset to know how many duplicate values exist

    # Check if column_subset is valid
    try:
        # Strip whitespaces
        if columns_subset: 
            columns_subset = [col.strip() for col in columns_subset]
            data[columns_subset]
    except (KeyError, TypeError):
        raise ValueError("The columns subset is not valid!")
    
    # Find duplicate values
        # keep='first' (default): Marks duplicates as True, except for the first occurrence.
        # keep='last': Marks duplicates as True, except for the last occurrence.
        # keep=False: Marks all duplicates (including the first and last) as True.
    data_duplicated = data.duplicated(keep=False, subset=columns_subset)
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling duplicate values.\nTop 10 of duplicate values are (Totally {data_duplicated.sum()} rows - including all duplicates, but from each group first one will remain and others will be removed):\n{data[data_duplicated].head(10)}")

    # Remove duplicate values
    # Subset is list of column names which we want to participate in the duplicate recognition
    # If it is None, all column values of a row should be the same as other's to consider as duplicates
    # here we use keep='first' (default), since we need to keep the first one from each group of duplicates
    data = data.drop_duplicates(subset=columns_subset)

    # Check dataset rows after removing duplicate rows
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling duplicate values.")

    return data


def handle_duplicate_values_fuzzy(data : pd.DataFrame, columns_subset : List = None, similarity_thresholds : Tuple = None, verbose : bool = False) -> pd.DataFrame:
    """
    Remove fuzzy (approximate) duplicate rows based on string similarity across specified columns.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to clean.
    columns_subset : list of str, optional
        Columns to use for similarity comparisons. 
        If None, all columns are used.
    similarity_thresholds : tuple (min_similarity: int, max_similarity: int), optional
        A tuple defining the range of average similarity scores (0-100) for considering two rows duplicates.
        Defaults to (90, 100).
    verbose : bool, default=False
        If True, displays detailed information including number of duplicates found, 
        sample outputs, and dataset size before and after removal.

    Returns
    -------
    pd.DataFrame
        A DataFrame with fuzzy duplicates removed, keeping the first occurrence in each group.

    Raises
    ------
    ValueError
        If any specified columns in `columns_subset` do not exist in the input DataFrame.

    Notes
    -----
    - The algorithm has O(n²) complexity; may be slow for datasets with many rows.
    - Preprocessing steps like lowercasing and stripping whitespace are automatically applied to strings.
    - Works best when applied to text/categorical columns.
    - If `similarity_thresholds = (100, 100)`, the behavior approximates exact duplicate detection.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # Note that if similarity_thresholds(100,100) is given to the function, the results are identical to handle_duplicate_values_drop() function

    # Check if column_subset is valid
    try:
        # Strip whitespaces
        if columns_subset: 
            columns_subset = [col.strip() for col in columns_subset]
            data[columns_subset]
    except (KeyError, TypeError):
        raise ValueError("The columns subset is not valid!")
    
    # If similarity_thresholds is not passed to the function it will be considered as (90,100)
    if similarity_thresholds is None:
        similarity_thresholds = (90,100)

    # If a subset of columns is given, we only consider these columns in similarity comparisons
    # If it is not assigned, we use all columns (It is better to give all categorical columns to the function, as the fuzz method is basically for string matching)
    comparison_columns = columns_subset if columns_subset else data.columns

    # This is a list containing sets of indexes. each set is for a group of duplicates.
    data_duplicated_sets = []
    # This is a list containing the similarity ratios of each column of under-comparison rows
    column_similarity_ratios = []
    # Iteration is on every unique non-ordered combination of the row indexes
    for i, j in combinations(data.index, 2):
        # For each comparison column, similarity ratio is calculated
        for col in comparison_columns:
            # The result is stored in ratios list
            column_similarity_ratios.append(fuzz.ratio(str(data.loc[i, col]).lower().strip(), str(data.loc[j, col]).lower().strip()))
        
        # # Average of similarity ratios of all columns is calculated.
        rows_similarity_avg_ratio = sum(column_similarity_ratios)/len(column_similarity_ratios)
        # If the result is in range, those rows will be considered as duplicates
        if similarity_thresholds[0] <= rows_similarity_avg_ratio <= similarity_thresholds[1]:
            # If it is the first group of duplicates, we add them without question
            if len(data_duplicated_sets) == 0:
                # Create a new set and add it to the list
                new_duplicated_set = set([i, j])
                data_duplicated_sets.append(new_duplicated_set)
            else:
                # It shows if we need to create a new set or add the indexes to the existing one
                new_set = True
                # We search if each item of our newly found pair is in the existing sets
                for d_set in data_duplicated_sets:
                    # If they exist, simply add both of them to the set
                    if i in d_set or j in d_set:
                        d_set.add(i)
                        d_set.add(j)
                        # No need to create a new set
                        new_set = False
                        break
                if new_set:
                    # If they don't exist, we need to create a new set of duplicates and add it to list
                    new_duplicated_set = set([i, j])
                    data_duplicated_sets.append(new_duplicated_set)
        
        # For each combination of rows, we need fresh ration list
        column_similarity_ratios.clear()

    # Based on the logic in handle_duplicate_values_drop() function, we show all the duplicated rows to the user (keep = False),
    # but should not eliminate the first duplicated row (keep='first')
    # So, need to have a set of indexes
    data_duplicated_set_show = set()
    data_duplicated_set_drop = set()
    for d_set in data_duplicated_sets:
        # Union all the sets of indexes (each group of duplicates) into one set, for further operations
        data_duplicated_set_show = data_duplicated_set_show.union(d_set)
        # Remove the first element of each group and union the others into one set
        first_duplicate = sorted(list(d_set))[0]
        d_set.remove(first_duplicate)
        data_duplicated_set_drop = data_duplicated_set_drop.union(d_set)

    # Convert sets to sorted lists
    data_duplicated_index_show = sorted(list(data_duplicated_set_show))
    data_duplicated_index_drop = sorted(list(data_duplicated_set_drop))

    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling duplicate values.\nTop 10 of duplicate values are (Totally {len(data_duplicated_index_show)} rows - including all duplicates, but from each group first one will remain and others will be removed):\n{data.iloc[data_duplicated_index_show].head(10)}")

    # Remove duplicate values
    data = data.drop(data_duplicated_index_drop)

    # Check dataset rows after removing duplicate rows
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling duplicate values.")

    return data


class HandleDuplicateStep(_PipelineStep):
    """
    Pipeline step for detecting and removing duplicate rows from a pandas DataFrame.

    Supports both exact and fuzzy duplicate handling depending on the specified method.

    Parameters
    ----------
    handle_duplicate_method : HandleDuplicateMethod
        The method to use: `EXACT` for full match, `FUZZY` for approximate matching.
    columns_subset : list of str, optional
        Specific columns to use for duplicate detection.
        If None, all columns are used.
    similarity_thresholds : tuple (min_similarity: int, max_similarity: int), optional
        Range of average similarity ratios for considering rows as fuzzy duplicates.
        Only applicable when `handle_duplicate_method` is `FUZZY`.
    verbose : bool, default=False
        If True, logs information on duplicates before and after removal.

    Methods
    -------
    apply (data: pd.DataFrame) -> pd.DataFrame:
        Apply the configured duplicate handling method to the input DataFrame.

    Notes
    -----
    - Designed for integration into custom data processing pipelines.
    - Fuzzy duplicate detection can significantly slow down processing on large datasets.
    """
    def __init__(self, 
                handle_duplicate_method : HandleDuplicateMethod,
                columns_subset : List = None,
                similarity_thresholds : Tuple = None, 
                verbose : bool = False
                ):
        self.handle_duplicate_method = handle_duplicate_method
        self.columns_subset = columns_subset
        self.similarity_thresholds = similarity_thresholds
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        match self.handle_duplicate_method:
            case HandleDuplicateMethod.EXACT:
                return handle_duplicate_values_exact(data, self.columns_subset, self.verbose)
            case HandleDuplicateMethod.FUZZY:
                return handle_duplicate_values_fuzzy(data, self.columns_subset, self.similarity_thresholds, self.verbose)
                
