import pandas as pd
import numpy as np
import pytest
from src.dataglass import *
from src.dataglass import DataPipeline


# Sample dataset to run the tests with
@pytest.fixture
def sample_data() -> pd.DataFrame:
# | index | name    | age  | income  | gender | country | score | signup_date | loyalty_score | subscription |
# |-------|---------|------|---------|--------|---------|-------|-------------|---------------|--------------|
# | 0     | Ethan   | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | 27            | Basic        |
# | 1     | Ethan   | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | 27            | Basic        |
# | 2     | Lili    | 22   | 45000   | Female | US      | 70    | 2023-03-01  | 25            | Basic        |
# | 3     | Sophia  | 40   | 80000   | Male   | DE      | 90    | 2023-04-01  | 24            | Premium      |
# | 4     | Mason   | 35   | 75000   | Female | FR      | 88    | 2023-05-01  | 23            | Basic        |
# | 5     | Ava     | 28   | 52000   | Male   | US      | 82    | 2023-06-20  | NaN           | Premium      |
# | 6     | Noah    | NaN  | 61000   | Female | UK      | 85    | 2023-07-01  | NaN           | Basic        |
# | 7     | Isabella| 32   | NaN     | Male   | DE      | 78    | 2023-08-01  | 20            | Premium      |
# | 8     | Lucas   | 27   | 50000   | NaN    | US      | 76    | 2023-09-01  | 19            | Basic        |
# | 9     | Mia     | 45   | 100000  | Female | FR      | NaN   | 2023-10-01  | 18            | Premium      |
# | 10    | James   | 29   | 58000   | Male   | US      | 79    | 2023-11-01  | 17            | Basic        |
# | 11    | Amelia  | 33   | 62000   | Female | DE      | 81    | 2023-12-01  | 16            | Premium      |
# | 12    | Benjamin| 24   | 47000   | Male   | UK      | 75    | 2024-01-01  | 15            | Basic        |
# | 13    | Jamey   | 55   | 200000  | Male   | US      | 95    | 2024-02-01  | 14            | Premium      |
# | 14    | Sophie  | 31   | 54000   | Female | FR      | 83    | 2024-03-01  | 13            | Basic        |
# | 15    | Harper  | 26   | 300000  | Female | US      | 200   | 2024-04-01  | 12            | Premium      |
# | 16    | Jack    | 60   | 49000   | Male   | UK      | 77    | 2024-05-01  | 11            | Basic        |
# | 17    | Evelyn  | 42   | 85000   | Female | DE      | 89    | 2024-06-01  | 10            | Premium      |
# | 18    | Alex    | 25   | 50000   | Male   | US      | 80    | 2025-01-01  | 9             | Basic        |
# | 19    | Lily    | 22   | 45000   | Female | DE      | 70    | 2025-04-01  | 8             | Basic        |


    contents = {
        "name": ["Ethan", "Ethan", "Lili", "Sophia", "Mason", "Ava", "Noah", "Isabella",
                 "Lucas", "Mia", "James", "Amelia", "Benjamin", "Jamey", "Sophie", "Harper", "Jack", "Evelyn", "Alex", "Lily"],
        "age": [25.0, 25, 22, 40, 35, 28, np.nan, 32, 27, 45,
                29, 33, 24, 55, 31, 26, 60, 42, 25, 22],
        "income": [50000, 50000, 45000, 80000, 75000, 52000, 61000, np.nan, 50000, 100000,
                58000, 62000, 47000, 200000, 54000, 300000, 49000, 85000, 50000, 45000],
        "gender": ["Male", "Male", "Female", "Male", "Female", "Male", "Female", "Male", np.nan, "Female",
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"],
        "country": ["US", "US", "US", "DE", "FR", "US", "UK", "DE", "US", "FR",
                    "US", "DE", "UK", "US", "FR", "US", "UK", "DE", "US", "DE"],
        "score": [80, 80, 70, 90, 88, 82, 85, 78, 76, np.nan,
                79, 81, 75, 95, 83, 200, 77, 89, 80, 70],
        "signup_date": [
            "2023-01-01", "2023-01-01", "2023-03-01", "2023-04-01", "2023-05-01",
            "2023-06-20", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
            "2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01",
            "2024-04-01", "2024-05-01", "2024-06-01", "2025-01-01", "2025-04-01"
        ],
        "loyalty_score" : [27, 27, 25, 24, 23, np.nan, np.nan, 20, 19,
                            18,17, 16, 15, 14, 13, 12, 11, 10, 9, 8],
        "subscription": ["Basic", "Basic", "Basic", "Premium", "Basic", "Premium", "Basic", "Premium", "Basic", "Premium",
                        "Basic", "Premium", "Basic", "Premiums", "Basic", "Premium", "Basic", "Premium", "Basic", "Basic"
        ]
    }
    data = pd.DataFrame(contents, index=range(20))
    return data


# ======================================= #
#      Handle Missing functions tests     #
# ======================================= #

# ----------------Drop------------------ #

def test_handle_missing_drop(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_drop(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of missing
    assert len(result) == len(input_data.dropna())


# ---------Datatype Imputation----------- #

def test_handle_missing_datatype_imputation_mean(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MEAN, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns "age" and "score" by mean
    assert result.loc[6,"age"] == input_data["age"].mean(skipna=True)
    assert result.loc[9,"score"] == input_data["score"].mean(skipna=True)
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert result.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]

def test_handle_missing_datatype_imputation_median(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MEDIAN, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns "age" and "score" by median
    assert result.loc[6,"age"] == input_data["age"].median(skipna=True)
    assert result.loc[9,"score"] == input_data["score"].median(skipna=True)
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert result.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]

def test_handle_missing_datatype_imputation_mode(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MODE, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns "age" and "score" by mode
    assert result.loc[6,"age"] == input_data["age"].mode(dropna=True)[0]
    assert result.loc[9,"score"] == input_data["score"].mode(dropna=True)[0]
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert result.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]


# ---------Adjacent Imputation----------- #

def test_handle_missing_adjacent_value_imputation_backward(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.BACKWARD, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns "age", "score", and "gender" by bfill
    assert result.loc[6,"age"] == result.loc[7,"age"]
    assert result.loc[9,"score"] == result.loc[10,"score"] 
    assert result.loc[8,"gender"] == result.loc[9,"gender"]

def test_handle_missing_adjacent_value_imputation_forward(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.FORWARD, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns "age", "score", and "gender" by ffill
    assert result.loc[6,"age"] == result.loc[5,"age"]
    assert result.loc[9,"score"] == result.loc[8,"score"] 
    assert result.loc[8,"gender"] == result.loc[8,"gender"]

def test_handle_missing_adjacent_value_imputation_interpolation_linear(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.INTERPOLATION_LINEAR, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(result) == len(input_data)
    # Check the imputation of the columns loyalty_score by linear interpolation
    assert result.loc[5,"loyalty_score"] == 22
    assert result.loc[6,"loyalty_score"] == 21

def test_handle_missing_adjacent_value_imputation_interpolation_time(sample_data):
    input_data = sample_data.copy()
    result = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.INTERPOLATION_TIME, "signup_date", verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated 
    assert len(result) == len(input_data)
    # Check the imputation of the columns loyalty_score by time-based interpolation
    # Convert the signup_date to pandas datetime for date delta calculations
    signup_dates = pd.to_datetime(input_data["signup_date"], format="%Y-%m-%d")
    # Date distance between the row before and the row after NaN area
    date_delta = (signup_dates[7] - signup_dates[4]).days
    # Loyalty score distance between the row before and the row after NaN area
    loyalty_score_delta = input_data.loc[7,"loyalty_score"] - input_data.loc[4,"loyalty_score"]
    # Calculate the expected results based on the time-based interpolation formula
    assert result.loc[5,"loyalty_score"] == result.loc[4,"loyalty_score"] + (signup_dates[5] - signup_dates[4]).days / date_delta * loyalty_score_delta
    assert result.loc[6,"loyalty_score"] == result.loc[4,"loyalty_score"] + (signup_dates[6] - signup_dates[4]).days / date_delta * loyalty_score_delta


# ======================================= #
#      Handle Duplicate functions tests   #
# ======================================= #

# ----------------Exact------------------ #

def test_handle_duplicate_exact_allcolumns(sample_data):
    input_data = sample_data.copy()
    result = handle_duplicate_values_exact(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1))
    assert len(result) == len(input_data.drop_duplicates())

def test_handle_duplicate_exact_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    columns_subset = ["gender", "country", "score"]
    result = handle_duplicate_values_exact(input_data, columns_subset=columns_subset, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1,18))
    assert len(result) == 18


# ----------------Fuzzy------------------ #

def test_handle_duplicate_fuzzy_allcolumns_similaritythreshold_default(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (90, 100)
    result = handle_duplicate_values_fuzzy(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1))
    assert len(result) == 19

def test_handle_duplicate_fuzzy_allcolumns_similaritythreshold_85_100(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (85, 100)
    result = handle_duplicate_values_fuzzy(input_data, similarity_thresholds=(85,100), verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1), (2, 19))
    assert len(result) == 18

def test_handle_duplicate_fuzzy_subsetcolumns_similaritythreshold_default(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (90, 100)
    columns_subset = ["gender", "country", "score"]
    result = handle_duplicate_values_fuzzy(input_data, columns_subset=columns_subset, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1,5,18), (4,14), (10,13), (11,17), (12,16))
    assert len(result) == 13

def test_handle_duplicate_fuzzy_subsetcolumns_similaritythreshold_70_99(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (85, 100)
    columns_subset = ["name"]
    result = handle_duplicate_values_fuzzy(input_data, columns_subset=columns_subset, similarity_thresholds=(70,99), verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (2,19), (3,14), (10,13))
    assert len(result) == 17


# ======================================= #
#      Handle Outlier functions tests     #
# ======================================= #

# =========== Detect Outliers =========== #

# ----------------IQR-------------------- #

def test_detect_outlier_iqr_allcolumns(sample_data):
    input_data = sample_data.copy()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.IQR)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers 
    assert outliers == {"age": [16], "income": [13, 15], "score": [15]}
    # Check some boundaries
    assert boundaries["age"] == (6.25, 56.25)
    assert boundaries["income"] == (8750, 118750)

def test_detect_outlier_iqr_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    columns_subset = ["score"]
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.IQR, 
                                           columns_subset=columns_subset)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers 
    assert outliers == {"score": [15]}
    # Check len of boundaries
    assert len(boundaries) == 1


# --------------ZSCORE------------------- #

def test_detect_outlier_zscore_allcolumns(sample_data):
    input_data = sample_data.copy()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ZSCORE)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers 
    assert outliers == {"income": [15], "score": [15]}
    # Check some boundaries
    assert boundaries["income"] == (pytest.approx(-112691.6,1e-1), pytest.approx(271954.8,1e-1))

def test_detect_outlier_zscore_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    columns_subset = ["score"]
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ZSCORE, 
                                           columns_subset=columns_subset)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers 
    assert outliers == {"score": [15]}
    # Check len of boundaries
    assert len(boundaries) == 1


# -----------Isolation Forest------------- #

def test_detect_outlier_isolationforest_allcolumns_contaminationrate_auto_wholerow(sample_data):
    input_data = sample_data.copy()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ISOLATION_FOREST)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers 
    assert outliers["age"] == [2, 13, 15, 16, 17, 19]
    # Check some boundaries -> 4 observing columns
    assert len(boundaries) == 4

def test_detect_outlier_isolationforest_subsetcolumns_contaminationrate_auto_wholerow(sample_data):
    input_data = sample_data.copy()
    # All numeric columns except "loyalty_score"
    columns_subset = input_data.drop(["loyalty_score"], axis="columns").select_dtypes(include="number").columns.to_list()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ISOLATION_FOREST, 
                                           columns_subset=columns_subset)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers
    assert outliers["age"] == [9, 13, 15, 16]
    # Check some boundaries -> 3 observing columns
    assert len(boundaries) == 3

def test_detect_outlier_isolationforest_subsetcolumns_contaminationrate_0_15_wholerow(sample_data):
    input_data = sample_data.copy()
    # All numeric columns except "loyalty_score"
    columns_subset = input_data.drop(["loyalty_score"], axis="columns").select_dtypes(include="number").columns.to_list()
    # It means I believe 15% of my data rows are outliers
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ISOLATION_FOREST, 
                                           columns_subset=columns_subset, contamination_rate=0.15)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers
    assert outliers["age"] == [13, 15, 16]
    # Check some boundaries -> 3 observing columns
    assert len(boundaries) == 3

def test_detect_outlier_isolationforest_subsetcolumns_contaminationrate_0_15_percolumn(sample_data):
    input_data = sample_data.copy()
    # All numeric columns except "loyalty_score"
    columns_subset = input_data.drop(["loyalty_score"], axis="columns").select_dtypes(include="number").columns.to_list()
    # It means I believe 15% of my data rows are outliers
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.ISOLATION_FOREST, 
                                           columns_subset=columns_subset, contamination_rate=0.15, per_column_detection=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the detected outliers
    assert outliers["age"] == [9, 13, 16]
    # Check some boundaries -> 3 observing columns
    assert len(boundaries) == 3


# -----------Local Outlier Factor------------- #

def test_detect_outlier_lof_allcolumns_contaminationrate_auto_n_neighbors_default_wholerow(sample_data):
    input_data = sample_data.copy()
    numeric_columns = input_data.select_dtypes(include="number").columns.to_list()
    # Local Outlier Factore does not work if there is NaN values in the data
    input_data[numeric_columns] = input_data[numeric_columns].fillna(input_data[numeric_columns].median())
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.LOCAL_OUTLIER_FACTOR)
    # Original data which has its NaN values filled with median should remain unchanged
    filledna_sample_data = sample_data.copy()
    filledna_sample_data[numeric_columns] = sample_data[numeric_columns].fillna(sample_data[numeric_columns].median())
    assert input_data.equals(filledna_sample_data)
    # Check the detected outliers 
    assert outliers == {}
    # Check some boundaries
    assert len(boundaries) == 0

def test_detect_outlier_lof_allcolumns_contaminationrate_0_15_n_neighbors_6_wholerow(sample_data):
    input_data = sample_data.copy()
    numeric_columns = input_data.select_dtypes(include="number").columns.to_list()
    # Local Outlier Factore does not work if there is NaN values in the data
    input_data[numeric_columns] = input_data[numeric_columns].fillna(input_data[numeric_columns].median())
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, 
                                           contamination_rate=0.15, n_neighbors=6)
    # Original data which has its NaN values filled with median should remain unchanged
    filledna_sample_data = sample_data.copy()
    filledna_sample_data[numeric_columns] = sample_data[numeric_columns].fillna(sample_data[numeric_columns].median())
    assert input_data.equals(filledna_sample_data)
    # Check the detected outliers 
    assert outliers["age"] == [9, 13, 15]
    # Check some boundaries
    assert len(boundaries) == 4

def test_detect_outlier_lof_subsetcolumns_contaminationrate_0_15_n_neighbors_6_wholerow(sample_data):
    input_data = sample_data.copy()
    numeric_columns = input_data.select_dtypes(include="number").columns.to_list()
    # Local Outlier Factore does not work if there is NaN values in the data
    input_data[numeric_columns] = input_data[numeric_columns].fillna(input_data[numeric_columns].median())
    # All numeric columns except "loyalty_score"
    columns_subset = input_data.drop(["loyalty_score"], axis="columns").select_dtypes(include="number").columns.to_list()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, 
                                           columns_subset=columns_subset, contamination_rate=0.15, n_neighbors=6)
    # Original data which has its NaN values filled with median should remain unchanged
    filledna_sample_data = sample_data.copy()
    filledna_sample_data[numeric_columns] = sample_data[numeric_columns].fillna(sample_data[numeric_columns].median())
    assert input_data.equals(filledna_sample_data)
    # Check the detected outliers 
    assert outliers["age"] == [9, 13, 15]
    # Check some boundaries
    assert len(boundaries) == 3

def test_detect_outlier_lof_subsetcolumns_contaminationrate_0_15_n_neighbors_6_percolumn(sample_data):
    input_data = sample_data.copy()
    numeric_columns = input_data.select_dtypes(include="number").columns.to_list()
    # Local Outlier Factore does not work if there is NaN values in the data
    input_data[numeric_columns] = input_data[numeric_columns].fillna(input_data[numeric_columns].median())
    # All numeric columns except "loyalty_score"
    columns_subset = input_data.drop(["loyalty_score"], axis="columns").select_dtypes(include="number").columns.to_list()
    outliers, boundaries = detect_outliers(input_data, detect_outlier_method=DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, 
                                           columns_subset=columns_subset, contamination_rate=0.15, n_neighbors=6, per_column_detection=True)
    # Original data which has its NaN values filled with median should remain unchanged
    filledna_sample_data = sample_data.copy()
    filledna_sample_data[numeric_columns] = sample_data[numeric_columns].fillna(sample_data[numeric_columns].median())
    assert input_data.equals(filledna_sample_data)
    # Check the detected outliers 
    assert outliers["income"] == [9, 13, 15]
    # Check some boundaries
    assert len(boundaries) == 3


# =========== Handle Outliers =========== #

def test_handle_outlier_drop(sample_data):
    input_data = sample_data.copy()
    outliers = {"age" : [13, 15]}
    result = handle_outliers(input_data, HandleOutlierMethod.DROP, outliers=outliers, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the outliers deleted
    assert len(result) == 18

def test_handle_outlier_replace_with_median(sample_data):
    input_data = sample_data.copy()
    outliers = {"age" : [13, 15]}
    result = handle_outliers(input_data, handle_outlier_method=HandleOutlierMethod.REPLACE_WITH_MEDIAN, outliers=outliers, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the outlier indices remained
    assert len(result) == len(input_data)
    # Check if the outlier replaced with median
    assert result.loc[15,"age"] == input_data["age"].median()

def test_handle_outlier_cap_with_boundaries(sample_data):
    input_data = sample_data.copy()
    outliers = {"age" : [13, 15]}
    boundaries = {"age" : [30, 40]}
    result = handle_outliers(input_data, handle_outlier_method=HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers=outliers, boundaries=boundaries, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check the outlier indices remained
    assert len(result) == len(input_data)
    # Check if the outlier replaced with boundaries. It depends on what side they're placed.
    assert result.loc[13,"age"] == 40
    assert result.loc[15,"age"] == 30


# ======================================= #
#     convert datatype functions tests    #
# ======================================= #

# -----------------Auto------------------ #

def test_type_conversion_auto(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    result = convert_datatype_auto(droppedna_input_data, verbose=True)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check the inferred data types
    assert pd.api.types.is_integer_dtype(result["age"])
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])

# --------------User Defined------------- #

def test_type_conversion_userdefined(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    convert_scenario =  {
        "column": ["age", "score", "signup_date"],
        "datatype": ["int", "float", "datetime"],
        "format": ["", "", "%Y-%m-%d"]
    }
    result = convert_datatype_userdefined(droppedna_input_data, convert_scenario=convert_scenario, verbose=True)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check the inferred data types
    assert pd.api.types.is_integer_dtype(result["age"])
    assert pd.api.types.is_float_dtype(result["score"])
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])


# ======================================= #
#     encode feature functions tests      #
# ======================================= #

# -------------Label Encoding------------ #

def test_encode_feature_label_encoding_allcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.LABEL_ENCODING)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    assert result["gender_encoded"].isin([0,1]).all()
    assert result["country_encoded"].isin(range(4)).all()

def test_encode_feature_label_encoding_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    columns_subset = ["gender"]
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.LABEL_ENCODING, columns_subset=columns_subset)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    assert result["gender_encoded"].isin([0,1]).all()
    # Check other columns are not encoded
    assert not "country_encoded" in result.columns

# -------------Onehot Encoding------------ #

def test_encode_feature_onehot_encoding_allcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.ONEHOT_ENCODING)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    assert result["gender_Male"].isin([0,1]).all()
    assert result["gender_Female"].isin([0,1]).all()
    assert result["country_US"].isin([0,1]).all()
    
def test_encode_feature_onehot_encoding_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    columns_subset = ["gender"]
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.ONEHOT_ENCODING, columns_subset=columns_subset)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    assert result["gender_Male"].isin([0,1]).all()
    assert result["gender_Female"].isin([0,1]).all()
    # Check other columns are not encoded
    assert not "country_US" in result.columns

# ----------------Hashing---------------- #

def test_encode_feature_hashing_allcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.HASHING)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    print(result.columns)
    assert result["gender_hash_0"].isin([-1,0,1]).all()
    assert result["country_hash_0"].isin([-1,0,1]).all()
    assert result["country_hash_1"].isin([-1,0,1]).all()
    
def test_encode_feature_hashing_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    columns_subset = ["gender"]
    result = encode_feature(droppedna_input_data, feature_encoding_method=FeatureEncodingMethod.HASHING, columns_subset=columns_subset)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if the encoded columns only contains valid values
    assert result["gender_hash_0"].isin([-1,0,1]).all()
    # Check other columns are not encoded
    assert not "country_hash_0" in result.columns


# ======================================= #
#      scale feature functions tests      #
# ======================================= #


def test_scale_feature_minmax_scaling_l2normalization_false(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    scaling_scenario = {
            "column": ["age", "income", "score"],
            "scaling_method": ["MINMAX_SCALING", "ROBUST_SCALING", "ZSCORE_STANDARDIZATION"]
    }
    result = scale_feature(droppedna_input_data,scaling_scenario=scaling_scenario)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if minmax scaling result is between 0 and 1
    assert all(result["age"] >= 0) and all(result["age"] <= 1)
    # Check if after robust scaling --> median(result) == 0 and iqr(result) == 1
    iqr_income = result["income"].quantile(0.75) - result["income"].quantile(0.25)
    median_income = result["income"].median()
    assert median_income == pytest.approx(0,1e-1)
    assert iqr_income == pytest.approx(1,1e-1)
    # Check if after zscore standardization --> mean(result) == 0 and str(result) == 1
    mean_score = result["score"].mean()
    std_score = result["score"].std()
    assert mean_score ==  pytest.approx(0, 1e-1)
    assert std_score == pytest.approx(1, 1e-1)


def test_scale_feature_minmax_scaling_l2normalization_true(sample_data):
    input_data = sample_data.copy()
    # NaN values are dropped because int dtype does not support NaN values
    droppedna_input_data = input_data.dropna()
    scaling_scenario = {
            "column": ["age", "income", "score"],
            "scaling_method": ["MINMAX_SCALING", "ROBUST_SCALING", "ZSCORE_STANDARDIZATION"]
    }
    result = scale_feature(droppedna_input_data,scaling_scenario=scaling_scenario, apply_l2normalization=True)
    # Original data should remain unchanged, in this case NaN values are dropped first
    assert input_data.dropna().equals(droppedna_input_data)
    # Check if minmax scaling result is between 0 and 1
    assert all(result["age"] >= 0) and all(result["age"] <= 1)
    # Check if after l2_normalization all numeric columns of the result shape a unit vector
    numeric_columns = input_data.select_dtypes(include="number").columns.to_list()
    # Calculate l2_norm of all samples
    l2_norms = (result[numeric_columns]**2).sum(axis="columns")
    assert np.allclose(l2_norms, 1, 1e-6)

# ======================================= #
#              Pipline Tests              #
# ======================================= #

def test_pipeline_simplestway(sample_data):
    handle_missing = HandleMissingStep(HandleMissingMethod.DROP)
    handle_duplicate = HandleDuplicateStep(HandleDuplicateMethod.EXACT)
    handle_outlier = HandleOutlierStep(DetectOutlierMethod.IQR, HandleOutlierMethod.DROP)
    encode_feature = EncodeFeatureStep(FeatureEncodingMethod.LABEL_ENCODING, ["gender"])
    scale_feature = ScaleFeatureStep({"column": ["score"], "scaling_method": ["MINMAX_SCALING"]})
    type_conversion = TypeConversionStep(ConvertDatatypeMethod.AUTO, verbose = True)

    dp =  DataPipeline(
        [
        handle_missing,
        handle_duplicate,
        type_conversion,
        handle_outlier,
        scale_feature,
        encode_feature,
        ]
    )

    input_data = sample_data.copy()
    result = dp.apply(input_data)

    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of missing (5), duplicate (1), and outliers (3) -> 20-(5+1+3) = 11
    assert len(result) == 11
    # Check auto type conversion
    assert pd.api.types.is_integer_dtype(result["age"])
    assert pd.api.types.is_datetime64_dtype(result["signup_date"])
    # Check feature "score" is scaled to [0,1]
    assert (result["score"] >= 0).all() and (result["score"] <= 1).all()
    # Check feature "gender" is correctly encoded
    assert result["gender_encoded"].isin([0,1]).all()