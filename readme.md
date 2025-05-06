# 🔍✨ dataglass

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
- **inputNon-destructive processing** (Any operation leaves the input data unchanged)

---

## 📦 Installation

```bash
pip install dataglass
```

---

## 📘 Usage Examples (Pipeline vs Functional)

### 🧩 Pipeline Approach (Simplest Configuration)
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

#### ❓ Missing Handling Module
This module provides multiple strategies to handle missing data though these functions:

- ***handle_missing_values_drop***: Drop-based strategy
    - Eliminate all rows that contain any NaN value.

- ***handle_missing_values_datatype_imputation***: Data type–aware imputation
    - Fill missing numeric values using the specified strategy: mean, median, or mode.
    - Fill missing categorical values with the first mode of each column.

- ***handle_missing_values_adjacent_value_imputation***: Value propagation or interpolation
    - Forward fill (ffill)
    - Backward fill (bfill)
    - Linear interpolation
    - Time-based interpolation (if datetime index is present)

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
    numerinumeric_datatype_imputation_methodc_method = dg.NumericDatatypeImputationMethod.MEAN,
    verbose = True
)

print(df_cleaned)
```

#### 📑 Duplicate Handling Module
This module provides two strategies to handle duplicate data though these functions:

- ***handle_duplicate_values_exact***: Removes exact duplicate rows
    - Optionally, a specific set of columns can be provided for duplicate analysis via columns_subset

- ***handle_duplicate_values_fuzzy***: Removes approximate (fuzzy) duplicates based on string similarity
    - Define the similarity threshold (e.g., 70–90%)
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
df_cleaned = handle_duplicate_values_fuzzy(
    data = df, 
    columns_subset=columns_subset, 
    similarity_thresholds=(70,100), 
    verbose=True)

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