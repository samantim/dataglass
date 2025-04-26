import pandas as pd
import numpy as np
import pytest
from ..preprocessing import *
from ..pipeline import DataPipeline

# Sample dataset to run the tests with
@pytest.fixture
def sample_data() -> pd.DataFrame:
    # | id | age  | income  | gender | country | score | signup_date | subscription |
    # |----|------|---------|--------|---------|-------|-------------|--------------|
    # | 1  | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | Basic        |
    # | 2  | 30   | 60000   | Female | UK      | 85    | 2023-02-01  | Premium      |
    # | 3  | 22   | 45000   | Female | US      | 70    | 2023-03-01  | Basic        |
    # | 4  | 40   | 80000   | Male   | DE      | 90    | 2023-04-01  | Premium      |
    # | 5  | 35   | 75000   | Female | FR      | 88    | 2023-05-01  | Basic        |
    # | 6  | 28   | 52000   | Male   | US      | 82    | 2023-06-01  | Premium      |
    # | 7  | NaN  | 61000   | Female | UK      | 85    | 2023-07-01  | Basic        |missing
    # | 8  | 32   | NaN     | Male   | DE      | 78    | 2023-08-01  | Premium      |missing
    # | 9  | 27   | 50000   | NaN    | US      | 76    | 2023-09-01  | Basic        |missing
    # | 10 | 45   | 100000  | Female | FR      | NaN   | 2023-10-01  | Premium      |missing
    # | 11 | 29   | 58000   | Male   | US      | 79    | 2023-11-01  | Basic        |
    # | 12 | 33   | 62000   | Female | DE      | 81    | 2023-12-01  | Premium      |
    # | 13 | 24   | 47000   | Male   | UK      | 75    | 2024-01-01  | Basic        |
    # | 14 | 55   | 200000  | Male   | US      | 95    | 2024-02-01  | Premium      |outlier candidate (income)
    # | 15 | 31   | 54000   | Female | FR      | 83    | 2024-03-01  | Basic        |
    # | 16 | 60   | 300000  | Female | US      | 99    | 2024-04-01  | Premium      |outlier candidate (income)
    # | 17 | 26   | 49000   | Male   | UK      | 77    | 2024-05-01  | Basic        |
    # | 18 | 42   | 85000   | Female | DE      | 89    | 2024-06-01  | Premium      |
    # | 19 | 25   | 50000   | Male   | US      | 80    | 2023-01-01  | Basic        |duplicate (id = 1)
    # | 20 | 40   | 80000   | Male   | DE      | 90    | 2023-04-01  | Premium      |duplicate (id = 4)

    contents = {
        "age": [25.0, 30, 22, 40, 35, 28, np.nan, 32, 27, 45,
                29, 33, 24, 55, 31, 60, 26, 42, 25, 40],
        "income": [50000, 60000, 45000, 80000, 75000, 52000, 61000, np.nan, 50000, 100000,
                58000, 62000, 47000, 200000, 54000, 300000, 49000, 85000, 50000, 80000],
        "gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", np.nan, "Female",
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Male"],
        "country": ["US", "UK", "US", "DE", "FR", "US", "UK", "DE", "US", "FR",
                    "US", "DE", "UK", "US", "FR", "US", "UK", "DE", "US", "DE"],
        "score": [80, 85, 70, 90, 88, 82, 85, 78, 76, np.nan,
                79, 81, 75, 95, 83, 99, 77, 89, 80, 90],
        "signup_date": [
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
            "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
            "2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01",
            "2024-04-01", "2024-05-01", "2024-06-01", "2023-01-01", "2023-04-01"
        ],
        "subscription": ["Basic", "Premium", "Basic", "Premium", "Basic", "Premium", "Basic", "Premium", "Basic", "Premium",
                        "Basic", "Premium", "Basic", "Premiums", "Basic", "Premium", "Basic", "Premium", "Basic", "Premium"]
    }
    data = pd.DataFrame(contents, index=range(1, 21))
    return data


# Test simplest and minimal way of using the pipeline
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

    input = sample_data.copy()
    cleaned_data = dp.apply(input)

    print(input["age"].dropna() % 1)

    print(input.dtypes)
    print(cleaned_data.dtypes)
    print(cleaned_data)
    print(cleaned_data.shape)
    
    # Original data should remain unchanged
    assert input.equals(sample_data)
    # Check for elimination of missing, duplicate, and outliers
    assert len(cleaned_data) == 12
    # Check auto type conversion
    assert pd.api.types.is_integer_dtype(cleaned_data["age"])
    assert pd.api.types.is_datetime64_dtype(cleaned_data["signup_date"])
    # Check feature "score" is scaled to [0,1]
    assert (cleaned_data["score"] >= 0).all() and (cleaned_data["score"] <= 1).all()
    # Check feature "gender" is correctly encoded
    assert cleaned_data["gender_encoded"].isin([0,1]).all()