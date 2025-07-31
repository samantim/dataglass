import pandas as pd
import numpy as np
import pytest
from src.dataglass import *
from src.dataglass.automation.auto_preprocessing import _get_datatypes, auto_preprocess_for_analysis
pd.set_option('future.no_silent_downcasting', True)


# Sample dataset to run the tests with
@pytest.fixture
def sample_data() -> pd.DataFrame:
# | index | name    | age  | income  | gender | country | score | signup_date | loyalty_score | subscription | married |
# |-------|---------|------|---------|--------|---------|-------|-------------|---------------|--------------|---------|
# | 0     | Ethan   | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | 27            | NaN          | True    |
# | 1     | etan    | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | 27            | NaN          | False   |
# | 2     | Lili    | 22   | 45000   | NaN    | US      | 70    | NaN         | 25            | NaN          | True    |
# | 3     | Sophia  | 40   | 80000   | Male   | DE      | 90    | 2023-04-01  | 24            | NaN          | False   |
# | 4     | Mason   | 35   | 75000   | Female | FR      | 88    | 2023-05-01  | 23            | NaN          | False   |
# | 5     | Ava     | 28   | 52000   | Male   | US      | 82    | 2023-06-20  | NaN           | Premium      | False   |
# | 6     | Noah    | NaN  | 61000   | Female | UK      | 85    | 2023-07-01  | NaN           | Basic        | NaN     |
# | 7     | Isabella| 32   | NaN     | Male   | DE      | 78    | 2023-09-01  | 20            | NaN          | True    |
# | 8     | Lucas   | 27   | 50000   | NaN    | US      | 76    | NaN         | 19            | NaN          | False   |
# | 9     | Mia     | 40   | 100000  | Female | FR      | NaN   | 2023-10-01  | 18            | Premium      | False   |
# | 10    | James   | 29   | 58000   | Male   | US      | 79    | 2023-11-01  | 17            | Basic        | False   |
# | 11    | Amelia  | 33   | 62000   | Female | DE      | 81    | 2023-12-01  | 16            | Premium      | False   |
# | 12    | Benjamin| 24   | 47000   | Male   | UK      | 75    | 2024-01-01  | 15            | Basic        | True    |
# | 13    | Jamey   | 41   | 200000  | Male   | US      | 95    | 2024-02-01  | 14            | Premium      | False   |
# | 14    | Sophie  | 31   | 54000   | Female | FR      | 83    | 2024-03-01  | 13            | Basic        | False   |
# | 15    | Harper  | 26   | 300000  | Female | US      | 200   | 2024-04-01  | 12            | Premium      | False   |
# | 16    | Jack    | 30   | 49000   | Male   | UK      | 77    | 2024-05-01  | 11            | Basic        | True    |
# | 17    | Evelyn  | 42   | 85000   | Female | DE      | 89    | 2024-06-01  | 10            | Premium      | False   |
# | 18    | Lily    | 22   | 45000   | Female | DE      | 70    | 2025-04-01  | 8             | Basic        | True    |
# | 19    | Lily    | 22   | 45000   | Female | DE      | 70    | 2025-04-01  | 8             | Basic        | True    |


    contents = {
        "name": ["Ethan", "Etan", "Lili", "Sophia", "Mason", "Ava", "Noah", "Isabella",
                 "Lucas", "Mia", "James", "Amelia", "Benjamin", "Jamey", "Sophie", "Harper", "Jack", "Evelyn", "Lily", "Lily"],
        "age": [25.0, 25, 22, 40, 35, 28, np.nan, 32, 27, 40,
                29, 33, 24, 41, 31, 26, 30, 42, 25, 22],
        "income": [50000, 50000, 45000, 80000, 75000, 52000, 61000, np.nan, 50000, 100000,
                58000, 62000, 47000, 200000, 54000, 300000, 49000, 85000, 45000, 45000],
        "gender": ["Male", "Male", np.nan, "Male", "Female", "Male", "Female", "Male", np.nan, "Female",
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Female", "Female"],
        "country": ["US", "US", "US", "DE", "FR", "US", "UK", "DE", "US", "FR",
                    "US", "DE", "UK", "US", "FR", "US", "UK", "DE", "DE", "DE"],
        "score": [80, 80, 70, 90, 88, 82, 85, 78, 76, np.nan,
                79, 81, 75, 95, 83, 200, 77, 89, 70, 70],
        "signup_date": [
            "2023-01-01", "2023-01-01", np.nan, "2023-04-01", "2023-05-01",
            "2023-06-20", "2023-07-01", "2023-09-01", np.nan, "2023-10-01",
            "2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01",
            "2024-04-01", "2024-05-01", "2024-06-01", "2025-04-01", "2025-04-01"
        ],
        "loyalty_score" : [27, 27, 25, 24, 23, np.nan, np.nan, 20, 19,
                            18,17, 16, 15, 14, 13, 12, 11, 10, 8, 8],
        "subscription": [np.nan, np.nan, np.nan, np.nan, np.nan, "Premium", "Basic", np.nan, np.nan, "Premium",
                        "Basic", "Premium", "Basic", "Premiums", "Basic", "Premium", "Basic", "Premium", "Basic", "Basic"],
        "empty_col" : [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                       np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        "married": [True, False, True, False, False, False, np.nan, True, False, False,
                        False, False, True, False, False, False, True, False, True, True],

    }
    data = pd.DataFrame(contents, index=range(20))
    return data


@pytest.fixture
def generate_test_data():
    def _generate(case: str):
        np.random.seed(42)

        if case == "univariate_zscore":
            return pd.DataFrame({
                "A": np.concatenate([np.random.normal(0, 1, 98), [8, 9]])
            })

        elif case == "univariate_iqr":
            return pd.DataFrame({
                "B": np.concatenate([np.random.exponential(scale=1, size=98), [15, 20]])
            })

        elif case == "multivariate_lof":
            base = np.random.normal(0, 1, size=(250, 3))
            base[0:5] += 10  # inject multivariate outliers
            return pd.DataFrame(base, columns=["X", "Y", "Z"])

        elif case == "multivariate_isolationforest":
            base = np.random.normal(0, 1, size=(15000, 30))
            base[0:10] += 15
            columns = [f"f{i}" for i in range(30)]
            return pd.DataFrame(base, columns=columns)

        elif case == "no_action":
            return pd.DataFrame({
                "C": np.concatenate([np.random.exponential(scale=1, size=198), [30, 35]])
            })

        else:
            raise ValueError("Unknown test case.")
    return _generate


# ============================================ #
#   Automatic Handle Missing functions tests   #
# ============================================ #

def test_auto_handle_missing(sample_data):
    input_data = sample_data.copy()

    # Get datatypes of all columns
    data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns = _get_datatypes(input_data)

    result = auto_handle_missing_values(input_data, data_types, numeric_columns, categorical_columns, datetime_columns, bool_columns, datetime_dependent_numeric_columns)

    # Check for elimination of empty columns
    assert "empty_col" not in result.columns

    # Check for filling datetime columns with default value "01-01-1900"
    assert result.loc[2, "signup_date"] == pd.Timestamp(year=1900, month=1, day=1)
    assert result.loc[8, "signup_date"] == pd.Timestamp(year=1900, month=1, day=1)

    # Check the datatype of column signup_date to be datatime64[ns]
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"].dtype)

    # Check categorical variable is filled with mode
    assert result.loc[2, "gender"] == input_data["gender"].mode(dropna=True)[0]

    # Check categorical variable is filled with "Empty"
    assert result.loc[4, "subscription"] == "Empty"

    # Check imputing boolean columns with False
    assert result.loc[6, "married"] == False

    # Check time-based interpolation for numeric time-dependet columns
    # Convert the signup_date to pandas datetime for date delta calculations
    signup_dates = pd.to_datetime(input_data["signup_date"], format="%Y-%m-%d")
    # Date distance between the row before and the row after NaN area
    date_delta = (signup_dates[7] - signup_dates[4]).days
    # Loyalty score distance between the row before and the row after NaN area
    loyalty_score_delta = input_data.loc[7,"loyalty_score"] - input_data.loc[4,"loyalty_score"]
    # Calculate the expected results based on the time-based interpolation formula
    assert result.loc[5,"loyalty_score"] == int(result.loc[4,"loyalty_score"] + (signup_dates[5] - signup_dates[4]).days / date_delta * loyalty_score_delta)
    assert result.loc[6,"loyalty_score"] == int(result.loc[4,"loyalty_score"] + (signup_dates[6] - signup_dates[4]).days / date_delta * loyalty_score_delta)

    # Check mean imputation for numeric time-independet columns
    assert result.loc[6, "age"] == int(input_data["age"].mean(skipna=True))

    # Check median imputation for numeric time-independet columns
    assert result.loc[7, "income"] == int(input_data["income"].median(skipna=True))
    assert result.loc[9, "score"] == int(input_data["score"].median(skipna=True))


# ============================================ #
#  Automatic Handle Duplicates functions tests #
# ============================================ #

def test_auto_handle_duplicates(sample_data):
    input_data = sample_data.copy()

    # Get datatypes of all columns
    _, _, categorical_columns, datetime_columns, _, _ = _get_datatypes(input_data)

    result = auto_handle_duplicates(input_data, categorical_columns, datetime_columns)

    # Check exact duplicate removed
    assert len(result[result["name"] == "Lily"]) == 1

    # Check fuzzy matched removed
    assert result[result["name"] == "Etan"].empty


# ============================================ #
#   Automatic Handle Outliers functions tests  #
# ============================================ #

def test_auto_handle_outliers_univariate_zscore(generate_test_data):
    input_data = generate_test_data("univariate_zscore")

    # Get datatypes of all columns
    data_types, numeric_columns, _, _, _, _ = _get_datatypes(input_data)

    result = auto_handle_outliers(input_data, numeric_columns, data_types)
    assert result["A"].max() < 8


def test_auto_handle_outliers_univariate_iqr(generate_test_data):
    input_data = generate_test_data("univariate_iqr")

    # Get datatypes of all columns
    data_types, numeric_columns, _, _, _, _ = _get_datatypes(input_data)

    result = auto_handle_outliers(input_data, numeric_columns, data_types)
    
    assert result["B"].max() < 20


def test_auto_handle_outliers_multivariate_lof_drop(generate_test_data):
    input_data = generate_test_data("multivariate_lof")
    
    # Get datatypes of all columns
    data_types, numeric_columns, _, _, _, _ = _get_datatypes(input_data)

    result = auto_handle_outliers(input_data, numeric_columns, data_types)
    
    assert result.shape[0] < input_data.shape[0]


def test_auto_handle_outliers_multivariate_isolationforest_drop(generate_test_data):
    input_data = generate_test_data("multivariate_isolationforest")
    
    # Get datatypes of all columns
    data_types, numeric_columns, _, _, _, _ = _get_datatypes(input_data)

    result = auto_handle_outliers(input_data, numeric_columns, data_types)
    
    assert result.shape[0] < input_data.shape[0]


def test_auto_handle_outliers_no_action(generate_test_data):
    input_data = generate_test_data("no_action")
    
    # Get datatypes of all columns
    data_types, numeric_columns, _, _, _, _ = _get_datatypes(input_data)

    result = auto_handle_outliers(input_data, numeric_columns, data_types)
    
    assert np.allclose(result["C"], input_data["C"])
    
# ============================================ #
#   Automatic Preprocess tests  #
# ============================================ #

def test_handle_missing_drop(sample_data):
    input_data = sample_data.copy()
    result = auto_preprocess_for_analysis(input_data, verbose=True)
    # Original data should remain unchanged
    assert input_data.equals(sample_data)
    # Check for eliminations
    assert result.shape[0] < input_data.shape[0]
    assert result.shape[1] < input_data.shape[1]
    # Check for missing value and outlier handling
    input_data.loc[9, "score"] = input_data["score"].median()
    assert result.loc[15, "score"] == int(input_data.drop(index=[1,19])["score"].median())