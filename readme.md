# 🧠 dataglass

**A modular and lightweight library for preprocessing, analysis, and modeling structured datasets in Python.**

`dataglass` provides an easy-to-use yet powerful framework to handle essential preprocessing tasks such as missing value handling, duplicate removal, outlier detection and management, feature encoding, type conversion, and feature scaling — all designed to integrate with custom pipeline workflows.

---

## 🚀 Preprocessing Features

- **Missing Value Imputation** (Drop, Imputation by Datatype: Mean/Median/Mode, Imputation by Adjacent Values: Forward/Backward Fill, Interpolation: Linear/Time-based)
- **Duplicate Detection & Removal** (Exact & Fuzzy Matching with Full and Partial Similarity Check)
- **Outlier Detection & Handling** (Detection: IQR/Z-Score/Isolation Forest/LOF, Handling: Drop/Replace with Median/Cap with Boundaries, Visualization: Before vs After Boxplots/Histogram)
- **Feature encoding** (Label Encoding/Onehot Encoding/Hashing)
- **Type Conversion** (Automatic Datatype Inferring/User-defined Type Conversion)
- **Feature Scaling** (Scaling: Min-Max/Z-Score/Robust Scaling, Normalization: L2 Normalization)
- **Pipeline Compatibility** (Custom pipeline interface for reusable workflows)

---

## 📦 Installation

```bash
pip install dataglass
Requires: pandas, numpy, scikit-learn, seaborn, matplotlib, rapidfuzz, category_encoders

