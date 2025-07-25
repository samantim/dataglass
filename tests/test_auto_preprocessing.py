import pandas as pd
import numpy as np
import pytest
from src.dataglass import *
from src.dataglass import DataPipeline
from src.dataglass.automation.auto_preprocessing import auto_handle_missing_values
pd.set_option('future.no_silent_downcasting', True)


# Sample dataset to run the tests with
@pytest.fixture
def sample_data() -> pd.DataFrame:
# | index | name    | age  | income  | gender | country | score | signup_date | loyalty_score | subscription | married |
# |-------|---------|------|---------|--------|---------|-------|-------------|---------------|--------------|---------|
# | 0     | Ethan   | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | 27            | NaN          | True    |
# | 1     | Ethan   | 25   | 50000   | Male   | US      | 80    | NaN         | 27            | NaN          | False   |
# | 2     | Lili    | 22   | 45000   | NaN    | US      | 70    | 2023-03-01  | 25            | NaN          | True    |
# | 3     | Sophia  | 40   | 80000   | Male   | DE      | 90    | 2023-04-01  | 24            | NaN          | False   |
# | 4     | Mason   | 35   | 75000   | Female | FR      | 88    | 2023-05-01  | 23            | NaN          | False   |
# | 5     | Ava     | 28   | 52000   | Male   | US      | 82    | 2023-06-20  | NaN           | Premium      | False   |
# | 6     | Noah    | NaN  | 61000   | Female | UK      | 85    | 2023-07-01  | NaN           | Basic        | NaN     |
# | 7     | Isabella| 32   | NaN     | Male   | DE      | 78    | 2023-09-01  | 20            | NaN          | True    |
# | 8     | Lucas   | 27   | 50000   | NaN    | US      | 76    | NaN         | 19            | NaN          | False   |
# | 9     | Mia     | 45   | 100000  | Female | FR      | NaN   | 2023-10-01  | 18            | Premium      | False   |
# | 10    | James   | 29   | 58000   | Male   | US      | 79    | 2023-11-01  | 17            | Basic        | False   |
# | 11    | Amelia  | 33   | 62000   | Female | DE      | 81    | 2023-12-01  | 16            | Premium      | False   |
# | 12    | Benjamin| 24   | 47000   | Male   | UK      | 75    | 2024-01-01  | 15            | Basic        | True    |
# | 13    | Jamey   | 55   | 200000  | Male   | US      | 95    | 2024-02-01  | 14            | Premium      | False   |
# | 14    | Sophie  | 31   | 54000   | Female | FR      | 83    | 2024-03-01  | 13            | Basic        | False   |
# | 15    | Harper  | 26   | 300000  | Female | US      | 200   | 2024-04-01  | 12            | Premium      | False   |
# | 16    | Jack    | 60   | 49000   | Male   | UK      | 77    | 2024-05-01  | 11            | Basic        | True    |
# | 17    | Evelyn  | 42   | 85000   | Female | DE      | 89    | 2024-06-01  | 10            | Premium      | False   |
# | 18    | Alex    | 25   | 50000   | Male   | US      | 80    | 2025-01-01  | 9             | Basic        | False   |
# | 19    | Lily    | 22   | 45000   | Female | DE      | 70    | 2025-04-01  | 8             | Basic        | True    |


    contents = {
        "name": ["Ethan", "Ethan", "Lili", "Sophia", "Mason", "Ava", "Noah", "Isabella",
                 "Lucas", "Mia", "James", "Amelia", "Benjamin", "Jamey", "Sophie", "Harper", "Jack", "Evelyn", "Alex", "Lily"],
        "age": [25.0, 25, 22, 40, 35, 28, np.nan, 32, 27, 45,
                29, 33, 24, 55, 31, 26, 60, 42, 25, 22],
        "income": [50000, 50000, 45000, 80000, 75000, 52000, 61000, np.nan, 50000, 100000,
                58000, 62000, 47000, 200000, 54000, 300000, 49000, 85000, 50000, 45000],
        "gender": ["Male", "Male", np.nan, "Male", "Female", "Male", "Female", "Male", np.nan, "Female",
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"],
        "country": ["US", "US", "US", "DE", "FR", "US", "UK", "DE", "US", "FR",
                    "US", "DE", "UK", "US", "FR", "US", "UK", "DE", "US", "DE"],
        "score": [80, 80, 70, 90, 88, 82, 85, 78, 76, np.nan,
                79, 81, 75, 95, 83, 200, 77, 89, 80, 70],
        "signup_date": [
            "2023-01-01", np.nan, "2023-03-01", "2023-04-01", "2023-05-01",
            "2023-06-20", "2023-07-01", "2023-09-01", np.nan, "2023-10-01",
            "2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01",
            "2024-04-01", "2024-05-01", "2024-06-01", "2025-01-01", "2025-04-01"
        ],
        "loyalty_score" : [27, 27, 25, 24, 23, np.nan, np.nan, 20, 19,
                            18,17, 16, 15, 14, 13, 12, 11, 10, 9, 8],
        "subscription": [np.nan, np.nan, np.nan, np.nan, np.nan, "Premium", "Basic", np.nan, np.nan, "Premium",
                        "Basic", "Premium", "Basic", "Premiums", "Basic", "Premium", "Basic", "Premium", "Basic", "Basic"],
        "empty_col" : [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                       np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        "married": [True, False, True, False, False, False, np.nan, True, False, False,
                        False, False, True, False, False, False, True, False, False, True],

    }
    data = pd.DataFrame(contents, index=range(20))
    return data


# ============================================ #
#   Automatic Handle Missing functions tests   #
# ============================================ #

def test_handle_missing_drop(sample_data):
    input_data = sample_data.copy()
    result = auto_handle_missing_values(input_data)

    # Original data should remain unchanged
    assert input_data.equals(sample_data)

    # Check for elimination of empty columns
    assert "empty_col" not in result.columns

    # Check for filling datetime columns with default value "01-01-1900"
    assert result.loc[1, "signup_date"] == pd.Timestamp(year=1900, month=1, day=1)
    assert result.loc[8, "signup_date"] == pd.Timestamp(year=1900, month=1, day=1)

    # Check the datatype of column signup_date to be datatime64[ns]
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"].dtype)

    # Check categorical variable is filled with mode
    assert result.loc[2, "gender"] == "Male"

    # Check categorical variable is filled with "Empty"
    assert result.loc[4, "subscription"] == "Empty"

    # Check imputing boolean columns with False
    assert result.loc[6, "married"] == False

