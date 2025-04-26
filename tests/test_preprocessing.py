import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from ..preprocessing import *
from ..pipeline import DataPipeline

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
# | 15    | Harper  | 60   | 300000  | Female | US      | 99    | 2024-04-01  | 12            | Premium      |
# | 16    | Jack    | 26   | 49000   | Male   | UK      | 77    | 2024-05-01  | 11            | Basic        |
# | 17    | Evelyn  | 42   | 85000   | Female | DE      | 89    | 2024-06-01  | 10            | Premium      |
# | 18    | Alex    | 25   | 50000   | Male   | US      | 80    | 2025-01-01  | 9             | Basic        |
# | 19    | Lily    | 22   | 45000   | Female | DE      | 70    | 2025-04-01  | 8             | Basic        |


    contents = {
        "name": ["Ethan", "Ethan", "Lili", "Sophia", "Mason", "Ava", "Noah", "Isabella",
                 "Lucas", "Mia", "James", "Amelia", "Benjamin", "Jamey", "Sophie", "Harper", "Jack", "Evelyn", "Alex", "Lily"],
        "age": [25.0, 25, 22, 40, 35, 28, np.nan, 32, 27, 45,
                29, 33, 24, 55, 31, 60, 26, 42, 25, 22],
        "income": [50000, 50000, 45000, 80000, 75000, 52000, 61000, np.nan, 50000, 100000,
                58000, 62000, 47000, 200000, 54000, 300000, 49000, 85000, 50000, 45000],
        "gender": ["Male", "Male", "Female", "Male", "Female", "Male", "Female", "Male", np.nan, "Female",
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"],
        "country": ["US", "US", "US", "DE", "FR", "US", "UK", "DE", "US", "FR",
                    "US", "DE", "UK", "US", "FR", "US", "UK", "DE", "US", "DE"],
        "score": [80, 80, 70, 90, 88, 82, 85, 78, 76, np.nan,
                79, 81, 75, 95, 83, 99, 77, 89, 80, 70],
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

def test_handle_missing_drop(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_drop(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of missing
    assert len(cleaned_data) == len(input_data.dropna())


def test_handle_missing_datatype_imputation_mean(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MEAN, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns "age" and "score" by mean
    assert cleaned_data.loc[6,"age"] == input_data["age"].mean(skipna=True)
    assert cleaned_data.loc[9,"score"] == input_data["score"].mean(skipna=True)
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert cleaned_data.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]


def test_handle_missing_datatype_imputation_median(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MEDIAN, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns "age" and "score" by median
    assert cleaned_data.loc[6,"age"] == input_data["age"].median(skipna=True)
    assert cleaned_data.loc[9,"score"] == input_data["score"].median(skipna=True)
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert cleaned_data.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]


def test_handle_missing_datatype_imputation_mode(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_datatype_imputation(input_data, NumericDatatypeImputationMethod.MODE, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns "age" and "score" by mode
    assert cleaned_data.loc[6,"age"] == input_data["age"].mode(dropna=True)[0]
    assert cleaned_data.loc[9,"score"] == input_data["score"].mode(dropna=True)[0]
    # Despite NumericDatatypeImputationMethod, categorical columns are always imputed by mode
    assert cleaned_data.loc[8,"gender"] == input_data["gender"].mode(dropna=True)[0]


def test_handle_missing_adjacent_value_imputation_backward(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.BACKWARD, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns "age", "score", and "gender" by bfill
    assert cleaned_data.loc[6,"age"] == cleaned_data.loc[7,"age"]
    assert cleaned_data.loc[9,"score"] == cleaned_data.loc[10,"score"] 
    assert cleaned_data.loc[8,"gender"] == cleaned_data.loc[9,"gender"]


def test_handle_missing_adjacent_value_imputation_forward(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.FORWARD, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns "age", "score", and "gender" by ffill
    assert cleaned_data.loc[6,"age"] == cleaned_data.loc[5,"age"]
    assert cleaned_data.loc[9,"score"] == cleaned_data.loc[8,"score"] 
    assert cleaned_data.loc[8,"gender"] == cleaned_data.loc[8,"gender"]


def test_handle_missing_adjacent_value_imputation_interpolation_linear(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.INTERPOLATION_LINEAR, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns loyalty_score by linear interpolation
    assert cleaned_data.loc[5,"loyalty_score"] == 22
    assert cleaned_data.loc[6,"loyalty_score"] == 21


def test_handle_missing_adjacent_value_imputation_interpolation_time(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_missing_values_adjacent_value_imputation(input_data, AdjacentImputationMethod.INTERPOLATION_TIME, "signup_date", verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check if no row is eliminated 
    assert len(cleaned_data) == len(input_data)
    # Check the imputation of the columns loyalty_score by time-based interpolation
    # Convert the signup_date to pandas datetime for date delta calculations
    signup_dates = pd.to_datetime(input_data["signup_date"], format="%Y-%m-%d")
    # Date distance between the row before and the row after NaN area
    date_delta = (signup_dates[7] - signup_dates[4]).days
    # Loyalty score distance between the row before and the row after NaN area
    loyalty_score_delta = input_data.loc[7,"loyalty_score"] - input_data.loc[4,"loyalty_score"]
    # Calculate the expected results based on the time-based interpolation formula
    assert cleaned_data.loc[5,"loyalty_score"] == cleaned_data.loc[4,"loyalty_score"] + (signup_dates[5] - signup_dates[4]).days / date_delta * loyalty_score_delta
    assert cleaned_data.loc[6,"loyalty_score"] == cleaned_data.loc[4,"loyalty_score"] + (signup_dates[6] - signup_dates[4]).days / date_delta * loyalty_score_delta


# ======================================= #
#      Handle Duplicate functions tests   #
# ======================================= #

def test_handle_duplicate_exact_allcolumns(sample_data):
    input_data = sample_data.copy()
    cleaned_data = handle_duplicate_values_exact(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1))
    assert len(cleaned_data) == len(input_data.drop_duplicates())

def test_handle_duplicate_exact_subsetcolumns(sample_data):
    input_data = sample_data.copy()
    subset_columns = ["gender", "country", "score"]
    cleaned_data = handle_duplicate_values_exact(input_data, subset=subset_columns, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1,18))
    assert len(cleaned_data) == 18

def test_handle_duplicate_fuzzy_allcolumns_similaritythreshold_default(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (90, 100)
    cleaned_data = handle_duplicate_values_fuzzy(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1))
    assert len(cleaned_data) == 19

def test_handle_duplicate_fuzzy_allcolumns_similaritythreshold_85_100(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (85, 100)
    cleaned_data = handle_duplicate_values_fuzzy(input_data, similarity_thresholds=(85,100), verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1), (2, 19))
    assert len(cleaned_data) == 18

def test_handle_duplicate_fuzzy_subsetcolumns_similaritythreshold_default(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (90, 100)
    subset_columns = ["gender", "country", "score"]
    cleaned_data = handle_duplicate_values_fuzzy(input_data, subset=subset_columns, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (0,1,5,18), (4,14), (10,13), (11,17), (12,16))
    assert len(cleaned_data) == 13

def test_handle_duplicate_fuzzy_subsetcolumns_similaritythreshold_70_99(sample_data):
    input_data = sample_data.copy()
    # Similarity threshold -> (85, 100)
    subset_columns = ["name"]
    cleaned_data = handle_duplicate_values_fuzzy(input_data, subset=subset_columns, similarity_thresholds=(70,99), verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of duplicates (index -> (2,19), (3,14), (10,13))
    print(cleaned_data)
    assert len(cleaned_data) == 17

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
    cleaned_data = dp.apply(input_data)

    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for elimination of missing (5), duplicate (1), and outliers (2) -> 20-(5+1+2) = 12
    assert len(cleaned_data) == 12
    # Check auto type conversion
    assert pd.api.types.is_integer_dtype(cleaned_data["age"])
    assert pd.api.types.is_datetime64_dtype(cleaned_data["signup_date"])
    # Check feature "score" is scaled to [0,1]
    assert (cleaned_data["score"] >= 0).all() and (cleaned_data["score"] <= 1).all()
    # Check feature "gender" is correctly encoded
    assert cleaned_data["gender_encoded"].isin([0,1]).all()