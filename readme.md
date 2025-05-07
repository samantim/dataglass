# 🧊 dataglass

**A modular and lightweight library for preprocessing, analysis, and modeling structured datasets in Python.**

`dataglass` provides an easy-to-use yet powerful framework to handle essential preprocessing tasks such as missing value handling, duplicate removal, outlier detection and management, feature encoding, type conversion, and feature scaling — all designed to integrate with custom pipeline workflows.

---

## 🚀 Preprocessing Features

- **Missing Value Imputation** (Drop, Imputation by Datatype: Mean/Median/Mode, Imputation by Adjacent Values: Forward/Backward Fill, Interpolation: Linear/Time-based)
- **Duplicate Detection & Removal** (Exact & Fuzzy Matching with Full and Partial Similarity Check)
- **Outlier Detection & Handling** (Detection: IQR/Z-Score/Isolation Forest/LOF, Handling: Drop/Replace with Median/Cap with Boundaries, Visualization: Before vs After Boxplot/Histogram)
- **Feature encoding** (Label Encoding/Onehot Encoding/Hashing)
- **Type Conversion** (Automatic Datatype Inferring/User-defined Type Conversion)
- **Feature Scaling** (Scaling: Min-Max/Z-Score/Robust Scaling, Normalization: L2 Normalization)
- **Pipeline Compatibility** (Custom pipeline interface for reusable workflows)
- **Non-destructive processing** (Any operation leaves the input data unchanged)

---

## 📦 Installation

```bash
pip install dataglass
```

---

## 📘 Usage Examples (Pipeline vs Functional)
There are two approaches to using the library features: the **pipeline architecture** and **standalone function** usage. The following examples cover both methods.

<br>

### 🧩 Pipeline Approach (Simplest Configuration)
Use this approach when you want a clean, modular, and reusable workflow for **end-to-end preprocessing**.
<br>

```python
# Importing the library and dependencies
import dataglass as dg
import pandas as pd
import numpy as np

# Creating a sample dataframe with a missing value and a categorical column
df = pd.DataFrame({
    "name": ["John", "Jane", "Jack"],
    "age": [40, np.nan, 50],
    "gender": ["male", "female", "male"]
})

# Step 1: Handle missing values by dropping rows that contain any missing value
handle_missing = dg.HandleMissingStep(dg.HandleMissingMethod.DROP)

# Step 2: Handle duplicates by removing exact duplicate rows
handle_duplicate = dg.HandleDuplicateStep(dg.HandleDuplicateMethod.EXACT)

# Step 3: Automatically detect and convert datatypes; verbose=True prints conversion logs
type_conversion = dg.TypeConversionStep(dg.ConvertDatatypeMethod.AUTO, verbose=True)

# Step 4: Detect outliers using IQR and remove them
handle_outlier = dg.HandleOutlierStep(dg.DetectOutlierMethod.IQR, dg.HandleOutlierMethod.DROP)

# Step 5: Scale the 'age' column using Min-Max scaling
scale_feature = dg.ScaleFeatureStep({"column": ["age"], "scaling_method": ["MINMAX_SCALING"]})

# Step 6: Encode the 'gender' column using label encoding
encode_feature = dg.EncodeFeatureStep(dg.FeatureEncodingMethod.LABEL_ENCODING, ["gender"])

# Create the pipeline by chaining all the preprocessing steps in the desired order
dp = dg.DataPipeline([
    handle_missing,
    handle_duplicate,
    type_conversion,
    handle_outlier,
    scale_feature,
    encode_feature,
])

# Apply the pipeline to the dataframe
df_cleaned = dp.apply(df)

# Display the cleaned and transformed dataframe
print(df_cleaned)
```

<br>

### ⚙️ Functional Approach
Use this approach when you need fine-grained control or quick one-off transformations on specific parts of your data.
<br>
#### ❓ Missing Handling Module
This module provides multiple strategies to handle missing data though these functions:

- ***handle_missing_values_drop***: Drop-based strategy
    - `Eliminate` all rows that contain any NaN value.

- ***handle_missing_values_datatype_imputation***: Data type–aware imputation
    - Fill missing numeric values using the specified strategy: `mean`, `median`, or `mode`.
    - Fill missing categorical values with the first `mode` of each column.

- ***handle_missing_values_adjacent_value_imputation***: Value propagation or interpolation
    - `Forward fill (ffill)`
    - `Backward fill (bfill)`
    - `Linear interpolation`
    - `Time-based interpolation` (if datetime index is present)

```python
import dataglass as dg
import pandas as pd
import numpy as np

# Creating a sample dataframe with a missing value
df = pd.DataFrame({
    "name": ["John", "Jane", "Jack"],
    "age": [40, np.nan, 50],
    "gender": ["male", "female", np.nan]
})

# Impute numeric columns using mean and the categorical columns using the first mode of that column
df_cleaned = dg.handle_missing_values_datatype_imputation(
    data = df,
    numeric_datatype_imputation_method = dg.NumericDatatypeImputationMethod.MEAN,
    verbose = True
)

print(df_cleaned)
```

#### 📑 Duplicate Handling Module
This module provides two strategies to handle duplicate data though these functions:

- ***handle_duplicate_values_exact***: Remove `exact duplicate` rows
    - Optionally, a specific set of columns can be provided for duplicate analysis via columns_subset

- ***handle_duplicate_values_fuzzy***: Remove `approximate (fuzzy) duplicates` based on string similarity
    - Define the `similarity threshold` (e.g., 70–90%)
    - Limit the comparison to specific columns via columns_subset


```python
import dataglass as dg
import pandas as pd
import numpy as np

# Creating a sample dataframe with a similar name values
df = pd.DataFrame({
    "name": ["John", "Johney", "Jack"],
    "age": [40, 45, 50],
})

# Only "name" column will be used to detect fuzzy duplicates
columns_subset = ["name"]

# Remove rows that are 70% or more similar in the "name" column (It keeps the first occurrence of each similarity group)
df_cleaned = dg.handle_duplicate_values_fuzzy(
    data = df, 
    columns_subset = columns_subset, 
    similarity_thresholds = (70,100), 
    verbose = True)

print(df_cleaned)
```

#### ❗ Outlier Handling Module
This module separates the detection and handling of outliers, giving you flexibility and control.

- ***detect_outliers***: Detects outliers using various statistical or model-based techniques:
    - `IQR`, `ZSCORE`, `ISOLATION_FOREST`, `LOCAL_OUTLIER_FACTOR`
    - An optional list of columns can be specified; otherwise, all numeric columns are used
    - Customization options like `contamination_rate` and `n_neighbors` available for model-based methods

- ***handle_outliers***: Applies the selected strategy to the detected outliers
    - `DROP`: Remove rows containing outliers
    - `REPLACE_WITH_MEDIAN`: Replace outlier values with their column median
    - `CAP_WITH_BOUNDARIES`: Clip outlier values to the inlier boundary limits (based on the detection method)


```python
import dataglass as dg
import pandas as pd
import numpy as np

# Sample dataset with an outlier in the "age" column
df = pd.DataFrame({
    "name": ["John", "Johney", "Jack"],
    "age": [40, 45, 200],
})

# Step 1: Detect outliers using the Z-Score method
outliers, boundaries = dg.detect_outliers(
    data = df, 
    detect_outlier_method = dg.DetectOutlierMethod.ZSCORE)

# Step 2: Cap outlier values with the calculated boundaries
df_cleaned = dg.handle_outliers(
    data = df,
    handle_outlier_method = dg.HandleOutlierMethod.CAP_WITH_BOUNDARIES,
    outliers = outliers,
    boundaries=boundaries,
    verbose=True)

print(df_cleaned)
```


#### 🔢 Feature Encoding Module
This module provides multiple methods to encode categorical features into numerical representations suitable for machine learning.

- ***encode_feature***: 
    - Supported methods: LABEL_ENCODING, ONEHOT_ENCODING, HASHING
    - Optionally specify columns; otherwise, all categorical columns will be encoded
    - To apply different methods to different columns, call the function multiple times with desired parameters

```python
import dataglass as dg
import pandas as pd
import numpy as np

# Sample dataset with a categorical "gender" column
df = pd.DataFrame({
    "name": ["John", "Jane", "Jack"],
    "age": [40, 45, 50],
    "gender": ["male", "female", "male"]
})

# Only "gender" column will be encoded
columns_subset = ["gender"]

# Convert "gender" to numerical labels (e.g., male=1, female=0)
df_cleaned = dg.encode_feature(
    data = df,
    feature_encoding_method = dg.FeatureEncodingMethod.LABEL_ENCODING,
    columns_subset = columns_subset)

print(df_cleaned)
```

#### 🔁 Type Conversion Module
This module provides methods for converting column datatypes for better compatibility and precision.

- ***convert_datatype_auto***: 
    - Automatically infers and converts column datatypes based on heuristics.
- ***convert_datatype_userdefined***:
    - Converts column datatypes based on a user-defined mapping scenario (supports formats like datetime parsing).

```python
import dataglass as dg
import pandas as pd
import numpy as np

# Sample dataset with mixed types
df = pd.DataFrame({
    "name": ["John", "Jane", "Jack"],
    "age": [40.0, 45, 50.0],
    "signup_date": ["2023-01-01", "2023-01-01", "2023-03-01"]
})

# user-defined scenario to request how to convert specific columns
convert_scenario =  {
    "column": ["age", "signup_date"],
    "datatype": ["int", "datetime"],
    "format": ["", "%Y-%m-%d"]
}

# Apply type conversion using the user-defined configuration
df_cleaned = dg.convert_datatype_userdefined(
    data = df,
    convert_scenario = convert_scenario,
    verbose=True)

print(df_cleaned)
```

#### 📏 Feature Scaling Module
This module allows feature scaling using different methods on selected columns, with an optional L2 normalization across all numeric columns.

- ***scale_feature***: 
    - Supported scaling methods: `MINMAX_SCALING`, `ZSCORE_STANDARDIZATION`, `ROBUST_SCALING`
    - L2 normalization can be optionally applied to all numeric columns after scaling
    - Scaling can be customized per column using the scaling_scenario

```python
import dataglass as dg
import pandas as pd
import numpy as np

# Sample dataset with numeric features
df = pd.DataFrame({
    "name": ["John", "Jane", "Jack"],
    "age": [40, 45, 50],
    "score": [60, 70, 180],
    "income": [5000, 4500, 3000]
})

# Define a scenario to scale "age" using MinMax and "score" using RobustScaler
scaling_scenario = {
    "column": ["age", "score"],
    "scaling_method": ["MINMAX_SCALING", "ROBUST_SCALING"]
}

# Apply scaling and then L2 normalize all numeric features
df_cleaned = dg.scale_feature(
    data = df,
    scaling_scenario = scaling_scenario,
    apply_l2normalization = True)

print(df_cleaned)
```

---

## ✅ Requirements

- Python ≥ 3.8  
All other dependencies will be installed automatically via `pip install dataglass`.

---

## 🛣️ Roadmap  

- ✅ Preprocessing Modules  
- ✅ Custom Pipelines  
- ⏳ Exploratory Data Analysis (EDA) 
- ⏳ Machine Learning Modules

---

## 📄 License  

This project is licensed under the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for more details.

---

## 🤝 Contributing  

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request via [GitHub](https://github.com/samantim/dataglass).

---

## 👤 Author  

**Saman Teymouri**  
*Data Scientist/Analyst & Python Developer*  
Berlin, Germany