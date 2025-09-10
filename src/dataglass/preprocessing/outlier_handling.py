"""
Outlier Detection and Handling Module
======================================

This module provides tools for detecting and managing outliers in pandas DataFrames, supporting both
traditional statistical techniques and machine learning-based methods. It is designed for modular integration
within preprocessing pipelines, including support for per-column and multivariate outlier detection.

Core Features:
--------------
1. **Outlier Detection Methods**
   - IQR (Interquartile Range)
   - Z-Score
   - Isolation Forest
   - Local Outlier Factor (LOF)

2. **Flexible Scope**
   - Apply to all numeric columns or a user-defined subset.
   - Supports both per-column and multivariate detection (for Isolation Forest and LOF).

3. **Outlier Handling Strategies**
   - Drop
   - Replace with Median
   - Cap with Inlier Boundaries

4. **Boundary Tracking**
   - Automatically calculates and stores lower/upper inlier thresholds.

5. **Visualization**
   - Visual comparison of distributions before and after outlier handling.

6. **Pipeline Integration**
   - Via `HandleOutlierStep`, compatible with pipeline interface `_PipelineStep`.

Enums:
------
- `DetectOutlierMethod`: Defines the algorithm for outlier detection.
- `HandleOutlierMethod`: Defines the strategy for outlier handling.

Functions:
----------
- `detect_outliers`: Performs detection and returns outlier indices and inlier boundaries.
- `handle_outliers`: Applies the chosen strategy to manage detected outliers.
- `visualize_outliers`: Generates visualizations for outlier impact.

Classes:
--------
- `HandleOutlierStep`: Pipeline-compatible class for automated outlier detection and handling.
"""
import pandas as pd
from enum import Enum
from os import path, makedirs
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from ..pipeline.pipeline import _PipelineStep

class DetectOutlierMethod(Enum):
    """
    Enumeration for supported outlier detection techniques.

    Attributes
    ----------
        IQR: int
            Uses the Interquartile Range rule to detect outliers.
        ZSCORE: int
            Identifies points outside 3 standard deviations from the mean.
        ISOLATION_FOREST: int
            Detects anomalies using tree-based model.
        LOCAL_OUTLIER_FACTOR: int
            Identifies local density deviations using LOF.
    """
    IQR = 1
    ZSCORE = 2
    ISOLATION_FOREST = 3
    LOCAL_OUTLIER_FACTOR = 4


class HandleOutlierMethod(Enum):
    """
    Enumeration for supported strategies to handle detected outliers.

    Attributes
    ----------
        DROP: int
            Remove rows containing any outliers.
        REPLACE_WITH_MEDIAN: int
            Replace outlier values with the column median.
        CAP_WITH_BOUNDARIES: int
            Clip outlier values to calculated lower/upper bounds.
    """
    DROP = 1
    REPLACE_WITH_MEDIAN = 2
    CAP_WITH_BOUNDARIES = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    """
    Determines the numeric columns to be used for outlier detection.

    This helper function validates and filters the user-specified subset of columns 
    (if provided), ensuring only numeric columns are considered for further analysis. 
    If no subset is given, all numeric columns in the DataFrame are selected.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame from which to select columns.
        columns_subset (List): A list of column names to use for outlier detection.
            If None or empty, all numeric columns in the DataFrame are selected.

    Returns
    ----------
        List: 
            A list of valid numeric column names to be used in outlier detection.

    Raises
    ----------
        ValueError: 
            If the specified subset includes non-numeric columns or invalid column names.
    """
    # Prepare observing columns

    # All numeric columns of the dataset
    numeric_columns = data.select_dtypes(include="number").columns.to_list()

    if columns_subset: 
        # Strip whitespaces
        columns_subset = [col.strip() for col in columns_subset]
        try:
            # Check if one of its columns does not exist in numeric columns
            if not all(col in numeric_columns for col in columns_subset):
                raise ValueError("The columns subset contains non-numeric columns!")
            else:
                # If there is a valid subset, it is considered as the observing columns
                observing_columns = columns_subset
        except:
            raise ValueError("The columns subset is not valid!")
    else:
        # if there is and empty or none columns_subset, all numeric columns will be considered as the observing columns
        observing_columns = numeric_columns

    return observing_columns    


def detect_outliers(data : pd.DataFrame, detect_outlier_method : DetectOutlierMethod, columns_subset : List = None, contamination_rate : float | str = "auto" , n_neighbors : int = 20, per_column_detection : bool = False) -> Tuple:
    """
    Detects outliers in specified numeric columns using the selected method.

    Parameters
    ----------
        data (pd.DataFrame): Input DataFrame.
        detect_outlier_method (DetectOutlierMethod): Outlier detection technique to use (IQR, ZSCORE, ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR).
        columns_subset (List, optional): Subset of columns to process. Defaults to all numeric columns.
        contamination_rate (float | str, optional): Expected proportion of outliers (used by Isolation Forest and LOF).
        n_neighbors (int, optional): Number of neighbors for LOF. Reasonable range: 25-30% of samples.
        per_column_detection (bool, optional): If True, applies model to each column independently. It is only recommended for experiment and comparison, since it does not make sense for real-world scenarios.

    Returns
    ----------
        Tuple (Dict, Dict)
            - outliers: Column-wise index lists of detected outliers.
            - boundaries: Column-wise (lower, upper) boundaries for inliers.

    Raises
    ----------
        ValueError
            If parameters are invalid or method cannot handle missing values.
    """
    # Parameter contamination_rate is used for training ISOLATION FOREST LOCAL OUTLIER FACTOR methods to set the boundaries for outliers
    # Parameter n_neighbors is used for training OUTLIER FACTOR methods to set the number of observing neighbors
    # Parameter per_column_detection is used for training ISOLATION FOREST and LOCAL OUTLIER FACTOR methods, as their main usage in analyzing multivariate data rather than univariates
    # but in case of comparison, I include both per-column and all-columns approaches for these methods

    # Check if column_subset is valid
    observing_columns = _get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return dict(), dict()

    # Validating contamination_rate
    if contamination_rate != "auto":
        try:
            contamination_rate = float(contamination_rate)
            if not (0.0 < contamination_rate <= 0.5):
                raise ValueError("The 'contamination_rate' parameter must be a float in the range (0.0, 0.5]")
        except (TypeError, ValueError):
            raise ValueError("The 'contamination_rate' parameter must be a str among {'auto'} or a float in the range (0.0, 0.5]")
    
    if detect_outlier_method == DetectOutlierMethod.LOCAL_OUTLIER_FACTOR:
        if data[observing_columns].isna().sum().sum() > 0:
            raise ValueError("Local outlier factor method does not work when the data contains NaN values.")

    outliers = {}
    boundaries = {}
    # Check detecting method and run the following block
    match detect_outlier_method:
        case DetectOutlierMethod.IQR:
            for col in observing_columns:
                # Calculate quantiles 1, 3
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                # Extract outliers based on IQR method
                outlier_indexes = data.loc[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].index.to_list()
                if len(outlier_indexes) > 0:
                    outliers[col] = outlier_indexes
                    boundaries[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        case DetectOutlierMethod.ZSCORE:
            for col in observing_columns:
                mean = data[col].mean()
                std = data[col].std()
                
                # Extract outliers based on Z-Score method
                # Z-Score = (data - mean)/std  -->  to be inlier  -->  -3 < Z-Score < 3
                outlier_indexes = data.loc[(data[col] < mean - 3 * std) | (data[col] > mean + 3 * std)].index.to_list()
                if len(outlier_indexes) > 0:
                    outliers[col] = outlier_indexes
                    boundaries[col] = (mean - 3 * std, mean + 3 * std)

        case DetectOutlierMethod.ISOLATION_FOREST:
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                    isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = isolation_forest.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outlier_indexes = data[predictions == -1].index.to_list()
                    if len(outlier_indexes) > 0:
                        outliers[col] = outlier_indexes
                        # Isolation forest method, rather than IQR and Z-Score, does not have native boundaries. So, we use the min and max value of inliers for that
                        inliers = data.loc[~data.index.isin(outliers[col]), col]
                        boundaries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                # Train the model based on all columns under observation
                predictions = isolation_forest.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outlier_indexes = data[predictions == -1].index.to_list()
                if len(outlier_indexes) > 0:
                    for col in observing_columns:
                        outliers[col] = outlier_indexes
                        # Isolation forest method, rather than IQR and Z-Score, does not have native boundaries. So, we use the min and max value of inliers for that
                        inliers = data.loc[~data.index.isin(outliers[col]), col]
                        boundaries[col] = (inliers.min(), inliers.max())

        case DetectOutlierMethod.LOCAL_OUTLIER_FACTOR:
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                    local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = local_outlier_factor.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outlier_indexes = data[predictions == -1].index.to_list()
                    if len(outlier_indexes) > 0:
                        outliers[col] = outlier_indexes
                        # Local outlier factor method, rather than IQR and Z-Score, does not have native boundaries. So, we use the min and max value of inliers for that
                        inliers = data.loc[~data.index.isin(outliers[col]), col]
                        boundaries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                # Train the model based on all columns under observation
                predictions = local_outlier_factor.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outlier_indexes = data[predictions == -1].index.to_list()
                if len(outlier_indexes) > 0:
                    for col in observing_columns:
                        outliers[col] = outlier_indexes
                        # Local outlier factor method, rather than IQR and Z-Score, does not have native boundaries. So, we use the min and max value of inliers for that
                        inliers = data.loc[~data.index.isin(outliers[col]), col]
                        boundaries[col] = (inliers.min(), inliers.max())
        case _:
            raise ValueError("Detect outlier method is not valid. It should be selected using DetectOutlierMethod enum.")
        
    # Output of the function is a Tuple consists of oulier indexes dict and boundaries on inliers dict
    return outliers, boundaries


def handle_outliers(data : pd.DataFrame, handle_outlier_method : HandleOutlierMethod, outliers : Dict = {}, boundaries : Dict = {}, verbose : bool = False) -> pd.DataFrame:
    """
    Handles previously detected outliers using a specified strategy.

    Parameters
    ----------
        data (pd.DataFrame): Input DataFrame.
        handle_outlier_method (HandleOutlierMethod): Strategy to apply (DROP, REPLACE_WITH_MEDIAN, CAP_WITH_BOUNDARIES).
        outliers (Dict): Dictionary mapping column names to outlier row indices.
        boundaries (Dict): Column-wise (lower, upper) values to cap outliers.
        verbose (bool): If True, prints diagnostics before and after processing.

    Returns
    ----------
        pd.DataFrame
            Cleaned DataFrame with handled outliers.

    Raises
    ----------
        KeyError
            If column names in `outliers` or `boundaries` are not present in the DataFrame.
    """
    # Display dataset info before and after imputation if verbose is enabled
    # If the outlier dict is empty, the output is the original data
    if len(outliers) == 0: return data

    if not all(col in data.columns for col in outliers.keys()):
        raise KeyError("At least one of the column names in the outliers dict is not valid.")
    
    if not all(col in data.columns for col in boundaries.keys()):
        raise KeyError("At least one of the column names in the boundaries dict is not valid.")

    # First unpack the values of the outlier dict and then union all of them in a set (to eliminate duplicate indexes)
    all_drop_indexes = set().union(*outliers.values())
    
    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling outliers values.\nTop 10 of rows containing outliers are (Totally {len(all_drop_indexes)} rows):\n{data.loc[sorted(all_drop_indexes)].head(10)}")

    match handle_outlier_method:
        case HandleOutlierMethod.DROP:
            # Drop all outliers
            result = data.drop(all_drop_indexes)
        case HandleOutlierMethod.REPLACE_WITH_MEDIAN:
            for col in outliers.keys():
                # For each column which has outliers, all the outliers replace with Median of that column
                result = data.copy()
                result.loc[outliers[col], col] = result[col].median()
        case HandleOutlierMethod.CAP_WITH_BOUNDARIES:
            for col in outliers.keys(): 
                # For each column which has outliers, all the outliers cap (clip) with boundry values of that column
                lower, upper = boundaries[col]
                result = data.copy()
                # If the boundaries are float, the column type should be converted to float (implicit casting is deprecated)
                # "isinstance" is safer than "type", since it also include numpy types
                if isinstance(lower, float) or isinstance(upper, float):
                    result[col] = result[col].astype(float)
                result.loc[outliers[col], col] = result.loc[outliers[col], col].clip(lower, upper)
        case _:
            raise ValueError("Handle outlier method is not valid. It should be selected using HandleOutlierMethod enum.")

    # Check dataset rows after removing duplicate rows
    if verbose:
        print(f"Dataset has {result.shape[0]} rows after handling outliers.")

    return result


def visualize_outliers(original_data : pd.DataFrame, cleaned_data : pd.DataFrame, output_dir : str, detect_outlier_method : DetectOutlierMethod, handle_outlier_method : HandleOutlierMethod, columns_subset : List = None):
    """
    Generates visual diagnostics for outlier detection and handling.

    Produces a series of side-by-side boxplots and histograms for each column before and after handling.

    Parameters
    ----------
        original_data (pd.DataFrame): Data before outlier handling.
        cleaned_data (pd.DataFrame): Data after outlier handling.
        output_dir (str): Directory where visualizations will be saved.
        detect_outlier_method (DetectOutlierMethod): Detection method (IQR, ZSCORE, ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR). It is only used in file naming.
        handle_outlier_method (HandleOutlierMethod): Handling method (DROP, REPLACE_WITH_MEDIAN, CAP_WITH_BOUNDARIES). It is only used in file naming.
        columns_subset (List, optional): Subset of columns to visualize. Defaults to all numeric.

    Returns
    ----------
        None
    """
    # Check if column_subset is valid
    observing_columns = _get_observing_columns(original_data, columns_subset)
    if len(observing_columns) == 0: return

    # Make the visualization directory path
    visualization_dir = path.join(output_dir, "visualizations")
    # Create the folder
    makedirs(visualization_dir, exist_ok=True)

    # Set the resolution and quality
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)   
    
    for col in observing_columns:
        # Create subplots for the current column
        ax1 = plt.subplot(2, 2, 1)
        sns.boxplot(original_data[col], ax=ax1)
        ax1.set_title("Box Plot - Before Cleaning")

        ax2 = plt.subplot(2, 2, 2)
        sns.histplot(original_data[col], kde=True, ax=ax2)
        ax2.set_title("Histogram - Before Cleaning")

        ax3 = plt.subplot(2, 2, 3)
        sns.boxplot(cleaned_data[col], ax=ax3)
        ax3.set_title("Box Plot - After Cleaning")

        ax4 = plt.subplot(2, 2, 4)
        sns.histplot(cleaned_data[col], kde=True, ax=ax4)
        ax4.set_title("Histogram - After Cleaning")

        # Save the file with proper dpi
        file_name = path.join(visualization_dir, "_".join([detect_outlier_method.name, handle_outlier_method.name, col]) + ".png")
        plt.savefig(fname=file_name, format="png", dpi=fig.dpi)

        plt.clf()
        
    plt.close()


class HandleOutlierStep(_PipelineStep):
    """
    Pipeline-compatible step for outlier detection and handling.

    Integrates outlier detection and handing within a pipeline. Supports multiple detection
    and handling methods with configurable scope and verbosity.

    Parameters
    ----------
        detect_outlier_method (DetectOutlierMethod): Method for outlier detection (IQR, ZSCORE, ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR).
        handle_outlier_method (HandleOutlierMethod): Strategy for handling detected outliers (DROP, REPLACE_WITH_MEDIAN, CAP_WITH_BOUNDARIES).
        columns_subset (List, optional): List of columns to inspect. Defaults to all numeric.
        contamination_rate (float | str, optional): Expected proportion of outliers (used by Isolation Forest and LOF).
        n_neighbors (int, optional): Number of neighbors for LOF. Reasonable range: 25-30% of samples.
        per_column_detection (bool, optional): If True, model is run per column (where applicable). It is only recommended for experiment and comparison, since it does not make sense for real-world scenarios.
        verbose (bool, optional): Enables detailed console output.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame:
            Executes the outlier detection and handling pipeline step.
    """
    def __init__(self, 
                detect_outlier_method : DetectOutlierMethod,
                handle_outlier_method : HandleOutlierMethod,
                columns_subset : List = None,
                contamination_rate : float | str = "auto",
                n_neighbors : int = 20,
                per_column_detection : bool = True,
                verbose : bool = False
                ):
    
        self.detect_outlier_method = detect_outlier_method
        self.handle_outlier_method = handle_outlier_method
        self.columns_subset = columns_subset
        self.contamination_rate = contamination_rate
        self.n_neighbors = n_neighbors
        self.per_column_detection = per_column_detection
        self.verbose = verbose

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        _outliers, _boundaries = detect_outliers(data, self.detect_outlier_method, self.columns_subset, self.contamination_rate, self.n_neighbors, self.per_column_detection)
        return handle_outliers(data, self.handle_outlier_method, _outliers, _boundaries,self.verbose)