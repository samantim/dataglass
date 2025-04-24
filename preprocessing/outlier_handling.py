import pandas as pd
from enum import Enum
from os import path, makedirs
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

class DetectOutlierMethod(Enum):
    IQR = 1
    ZSCORE = 2
    ISOLATION_FOREST = 3
    LOCAL_OUTLIER_FACTOR = 4


class HandleOutlierMethod(Enum):
    DROP = 1
    REPLACE_WITH_MEDIAN = 2
    CAP_WITH_BOUNDARIES = 3


def _get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    # Prepare observing columns
    # Strip whitespaces
    if columns_subset: columns_subset = [col.strip() for col in columns_subset]
    try:
        # If columns_subset only has numeric columns is valid
        numeric_columns = data.select_dtypes(include="number").columns
        # If columns_subset is not None and one of its columns does not exist in numeric columns
        if columns_subset and not all(col in numeric_columns for col in columns_subset):
            raise ValueError("The columns subset contains non-numeric columns!")
        else:
            # If there is a valid subset, it is considered as the observing columns otherwise all nemuric columns are considered
            observing_columns = columns_subset if columns_subset else numeric_columns
    except:
        raise ValueError("The columns subset is not valid!")

    return observing_columns    


def detect_outliers(data : pd.DataFrame, detect_outlier_method : DetectOutlierMethod, columns_subset : List = None, contamination_rate : float | str = "auto" , n_neighbors : int = 20, per_column_detection : bool = True) -> Tuple:
    """
    Detects outliers in the specified numeric columns using one of the supported methods.

    Args:
        data (pd.DataFrame): The input DataFrame.
        detect_outlier_method (DetectOutlierMethod): Method to use for detecting outliers.
        columns_subset (List, optional): List of columns to check. If None, all numeric columns are considered.
        contamination_rate (float | str, optional): Proportion of outliers in the data. Used in Isolation Forest and LOF.
        n_neighbors (int, optional): Number of neighbors to use for LOF.
        per_column_detection (bool, optional): If True, applies model per column; else applies on multivariate data.

    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - outliers (dict): Column-wise index lists of detected outliers.
            - boundaries (dict): Column-wise (lower, upper) boundaries for inliers.
    
    Raises:
        ValueError: If parameters are invalid.
    """

    # Parameter contamination_rate is used for training ISOLATION FOREST LOCAL OUTLIER FACTOR methods to set the boundries for outliers
    # Parameter n_neighbors is used for training OUTLIER FACTOR methods to set the number of observing neighbors
    # Parameter per_column_detection is used for training ISOLATION FOREST LOCAL OUTLIER FACTOR methods, as their main usage in analyzing multivariate data rather than univariates
    # but in case of comparison, I include both per-column and all-columns approaches for these methods

    # Check if column_subset is valid
    observing_columns = _get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return dict(), dict()

    # Validating contamination_rate
    if contamination_rate != "auto":
        try:
            contamination_rate = float(contamination_rate)
            if not (0.0 < contamination_rate <= 0.5):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("The 'contamination' parameter of IsolationForest must be a str among {'auto'} or a float in the range (0.0, 0.5]")
        
    outliers = {}
    boundries = {}
    # Check detecting method and run the following block
    match detect_outlier_method:
        case DetectOutlierMethod.IQR:
            for col in observing_columns:
                # Calculate quantiles 1, 3
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                # Extract outliers based on IQR method
                outliers[col] = data.loc[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].index.to_list()
                boundries[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        case DetectOutlierMethod.ZSCORE:
            for col in observing_columns:
                mean = data[col].mean()
                std = data[col].std()
                
                # Extract outliers based on Z-Score method
                # Z-Score = (data - mean)/std  -->  to be inlier  -->  -3 < Z-Score < 3
                outliers[col] = data.loc[(data[col] < mean - 3 * std) | (data[col] > mean + 3 * std)].index.to_list()
                boundries[col] = (mean - 3 * std, mean + 3 * std)

        case DetectOutlierMethod.ISOLATION_FOREST:
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                    isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = isolation_forest.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outliers[col] = data[predictions == -1].index.to_list()
                    # Isolation forest method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                # Train the model based on all columns under observation
                predictions = isolation_forest.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outliers_indexes = data[predictions == -1].index.to_list()
                for col in observing_columns:
                    outliers[col] = outliers_indexes
                    # Isolation forest method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())

        case DetectOutlierMethod.LOCAL_OUTLIER_FACTOR:
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                    local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = local_outlier_factor.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outliers[col] = data[predictions == -1].index.to_list()
                    # Local outlier factor method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                # Train the model based on all columns under observation
                predictions = local_outlier_factor.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outliers_indexes = data[predictions == -1].index.to_list()
                for col in observing_columns:
                    outliers[col] = outliers_indexes
                    # Local outlier factor method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())

    # Output of the function is a Tuple consists of oulier indexes dict and boundries on inliers dict
    return outliers, boundries


def handle_outliers(data : pd.DataFrame, handle_outlier_method : HandleOutlierMethod, outliers : Dict = {}, boundries : Dict = {}, verbose : bool = False) -> pd.DataFrame:
    """
    Handles the detected outliers in the DataFrame using the selected strategy.

    Args:
        data (pd.DataFrame): The input DataFrame.
        handle_outlier_method (HandleOutlierMethod): Strategy to handle outliers.
        outliers (Dict): Dictionary of detected outlier indices per column.
        boundries (Dict): Boundaries used to clip outliers if needed.
        verbose (bool): If True, prints dataset stats before and after handling.

    Returns:
        pd.DataFrame: DataFrame with outliers handled as specified.
    """
        
    # Display dataset info before and after imputation if verbose is enabled
    # If the outlier dict is empty, the output is the original data
    if len(outliers) == 0: return data

    # First unpack the values of the outlier dict and then union all of them in a set (to eliminate duplicate indexes)
    all_drop_indexes = set().union(*outliers.values())
    
    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    if verbose:
        print(f"Dataset has {data.shape[0]} rows before handling outliers values.\nTop 10 of rows containing outliers are (Totally {len(all_drop_indexes)} rows):\n{data.iloc[list(all_drop_indexes)].head(10)}")

    match handle_outlier_method:
        case HandleOutlierMethod.DROP:
            # Drop all outliers
            data = data.drop(all_drop_indexes)
        case HandleOutlierMethod.REPLACE_WITH_MEDIAN:
            for col in outliers.keys():
                # For each column which has outliers, all the outliers replace with Median of that column
                data.loc[outliers[col], col] = data[col].median()
        case HandleOutlierMethod.CAP_WITH_BOUNDARIES:
            for col in outliers.keys(): 
                # For each column which has outliers, all the outliers cap (clip) with boundry values of that column
                lower, upper = boundries[col]
                # If the boundries are float, the column type should be converted to float (implicit casting is deprecated)
                # "isinstance" is safer than "type", since it also include numpy types
                if isinstance(lower, float) or isinstance(upper, float):
                    data[col] = data[col].astype(float)
                data.loc[outliers[col], col] = data.loc[outliers[col], col].clip(lower, upper)

    # Check dataset rows after removing duplicate rows
    if verbose:
        print(f"Dataset has {data.shape[0]} rows after handling outliers.")

    return data


def visualize_outliers(original_data : pd.DataFrame, cleaned_data : pd.DataFrame, output_dir : str, detect_outlier_method : DetectOutlierMethod, handle_outlier_method : HandleOutlierMethod, columns_subset : List = None):
    """
    Visualizes the column-wise distribution of original and cleaned columns using boxplots and histograms.

    Args:
        original_data (pd.DataFrame): Data before outlier handling.
        cleaned_data (pd.DataFrame): Data after outlier handling.
        output_dir (str): Directory to save the visualizations. It will create a visualizations folder inside it.
        detect_outlier_method (DetectOutlierMethod): Method used for detecting outliers (used in filenames).
        handle_outlier_method (HandleOutlierMethod): Method used for handling outliers (used in filenames).
        columns_subset (List, optional): Columns to visualize. If None, all numeric columns are visualized.

    Returns:
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
        plt.subplot(2,2,1)
        sns.boxplot(original_data[col])

        plt.subplot(2,2,2)
        sns.histplot(original_data[col], kde=True)

        plt.subplot(2,2,3)
        sns.boxplot(cleaned_data[col])

        plt.subplot(2,2,4)
        sns.histplot(cleaned_data[col], kde=True)

        # Save the file with proper dpi
        file_name = path.join(visualization_dir, "_".join([detect_outlier_method.name, handle_outlier_method.name, col]) + ".png")
        plt.savefig(fname=file_name, format="png", dpi=fig.dpi)

        plt.clf()
        
    plt.close()