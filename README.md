# Task 21: Missing Value Analysis

## Description
This project focuses on identifying missing values within a dataset and calculating both the total missing counts and percentages for each column. Identifying and understanding missing data is a critical exploratory data analysis (EDA) step before building machine learning models.

---

## Objectives
* Understand the importance of detecting missing values prior to model training.
* Calculate total counts and percentages of missing values across all features.
* Identify which columns require specific treatment or imputation strategies.

---

## Tools & Libraries
* **Python**
* **Pandas**
* **Jupyter Notebook**

---

## Dataset Overview & Results

### **Summary Table**
* **Total Rows:** 891
* **Total Columns:** 15

| Column Name | Missing Count | Missing Percentage (%) |
| :--- | :---: | :---: |
| **deck** | 688 | 77.22% |
| **age** | 177 | 19.87% |
| **embarked** | 2 | 0.22% |
| **embark_town** | 2 | 0.22% |

---

## Key Insights & Action Plan

1. **`deck` (77.22% Missing):**
   * **Observation:** More than three-quarters of the data is missing.
   * **Action Plan:** Dropping this column or converting it into a binary feature indicating presence/absence (`deck_known`: 1 or 0) is recommended to prevent noise in model training.

2. **`age` (19.87% Missing):**
   * **Observation:** Nearly 20% of the age values are missing.
   * **Action Plan:** Impute missing values using numerical strategies, such as median imputation or group-wise median imputation based on passenger class (`pclass`) and sex (`sex`).

3. **`embarked` & `embark_town` (0.22% Missing):**
   * **Observation:** Only 2 rows have missing categorical values.
   * **Action Plan:** Impute using the statistical mode (most frequent categorical value) or simply drop the 2 missing rows.

---

## Python Implementation

```python
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

print("--- Dataset Overview ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

# Calculate counts and percentages of missing values
missing_counts = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)

# Build summary DataFrame
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage (%)': missing_percentage
})

# Display columns with missing values sorted descending
missing_summary_filtered = missing_summary[missing_summary['Missing Count'] > 0].sort_values(
    by='Missing Count', ascending=False
)

print("--- Missing Values Summary ---")
print(missing_summary_filtered)
```

---

## Interview Questions & Answers

### **Q1: What is a missing value?**
> **Answer:** A missing value occurs when no data is stored or recorded for a variable in an observation. In Python (Pandas), missing values are typically represented as `NaN` (Not a Number), `None`, or `NaT` (Not a Time).

---

### **Q2: Why can missing values be a problem in machine learning?**
> **Answer:**
> 1. **Algorithm Failure:** Most standard machine learning models in `scikit-learn` cannot handle missing values and will raise errors during model fitting.
> 2. **Model Bias:** Imputing or dropping missing values incorrectly can skew the distribution of features and introduce statistical bias.
> 3. **Information Loss:** Discarding entire rows due to missing values reduces dataset size and discards potentially useful signal from non-missing variables.

---

### **Q3: How can you identify missing values in Pandas?**
> **Answer:**
> * `df.isnull()` or `df.isna()`: Returns a boolean mask where `True` indicates missing values.
> * `df.isnull().sum()`: Sums boolean values column-wise to yield total missing counts per feature.
> * `df.info()`: Displays non-null counts per column.
> * `df.isnull().mean() * 100`: Calculates the percentage of missing values per column.
