# CAS Registry Numbers as a Pydantic baseclass and extended into a Pandas dtype

A Python utility class for validating Chemical Abstract Service (CAS) Registry Numbers®. Pydantic is used for validation and is a dependency.

## Overview

The CAS class provides functionality to work with CAS Registry Numbers® (CAS RN®), which are unique numerical identifiers assigned to chemical substances. This tool helps validate the format and checksum of CAS RNs and enables sorting collections of CAS numbers and use in Pandas.

## Features

- Validates Chemical Abstract Service Registry Number® (CAS RN®) format
- Verifies CAS RN checksum
- Makes CAS numbers sortable in ascending or descending order based on numeric comparison rather than string-based comparison
- CAS object instances are hashed which allows for uses in Python Sets (i.e. a means to eliminate duplicates)
- CAS object instances are __str__ printed as plain strings of the CAS formatted registry number
- CAS object instances can be JSON serialized by using Pydantic .model_dump_json()
- Pandas extension dtype (CASDtype) for working with CAS numbers in DataFrames with full type safety and validation

## Installation (Basic)

With uv (recommended):

```bash
uv add cas-reg
```

With pip:

```bash
pip install cas-reg
```

## Usage

```python
from cas_reg import CAS

# Create a CAS instance
cas_number = CAS(num="7732-18-5")  # Water
print(cas_number)
>>> "7732-18-5"

# use pydantic's by_alias to produce the model as a dictionary with CAS
print(cas_number.model_dump_json(by_alias=True))
>>> "{'CAS': '7732-18-5'}"


# Invalid CAS numbers do not validate
try:
    cas_string = "50-01-1"
    my_cas_no = CAS(num=cas_string)
except ValueError as e:
    print(f"CAS Number, {cas_string}, is invalid: {e}") 


# Sort multiple CAS numbers
cas_str_list = ["7732-18-5", "67-56-1", "124-38-9"]
cas_list = [CAS(num=x) for x in cas_str_list]
cas_list.sort()
print([str(x) for x in cas_list])  
>>> ["67-56-1", "124-38-9", "7732-18-5"]

# Use python sets to eliminate duplicates
cas_str_list2 = ["50-00-0", "7732-18-5", "67-56-1", "124-38-9", "50-00-0"]
cas_list2 = [CAS(num=x) for x in cas_str_list2]
unique_cas_list = list(set(cas_list2))
unique_cas_list.sort()
print([str(x) for x in unique_cas_list])
>>> ["50-00-0", "67-56-1", "124-38-9", "7732-18-5"]

```

## Pandas extension

The `cas-reg` package optionally provides a custom pandas extension dtype (`CASDtype`) and array type (`CASArray`) that allows you to work with CAS Registry Numbers in pandas DataFrames with full type safety and validation.

### Installation to support use in Pandas

```bash
# With uv
uv add "cas-reg[pandas]"

# With pip
pip install "cas-reg[pandas]"
```

### Dependencies for use with pandas

For pandas support the following packages are also dependencies and will be installed:

- pandas 2.0.0+
- numpy 1.24.0+

### Creating DataFrames with CAS columns

```python
import pandas as pd
from cas_reg import CAS
from cas_reg.pandas_ext import CASDtype, CASArray

# Create a DataFrame with CAS numbers directly
df = pd.DataFrame({
    "cas": CASArray(["50-00-0", "58-08-2", "7732-18-5"]),
    "name": ["Formaldehyde", "Caffeine", "Water"],
    "molecular_weight": [30.03, 194.19, 18.02]
})

print(df)
#        cas            name  molecular_weight
# 0  50-00-0   Formaldehyde              30.03
# 1  58-08-2        Caffeine             194.19
# 2  7732-18-5         Water              18.02

print(df["cas"].dtype)
# CAS
```

### Converting existing DataFrames

If you have an existing DataFrame with CAS numbers stored as strings (object dtype), you can convert them to the CASDtype:

```python
# Existing DataFrame with CAS numbers as strings
df = pd.DataFrame({
    "cas": ["50-00-0", "58-08-2", "7732-18-5"],
    "name": ["Formaldehyde", "Caffeine", "Water"]
})

print(df["cas"].dtype)  # object

# Convert the column to CASDtype
df["cas"] = df["cas"].astype(CASDtype())

print(df["cas"].dtype)  # CAS

# Each value is now a CAS object
print(type(df["cas"].iloc[0]))  # <class 'cas_reg.CAS'>
```

### Sorting DataFrames by CAS numbers

CAS numbers can be sorted naturally within a DataFrame:

```python
df = pd.DataFrame({
    "cas": CASArray(["58-08-2", "50-00-0", "7732-18-5", "67-56-1"]),
    "name": ["Caffeine", "Formaldehyde", "Water", "Methanol"]
})

# Sort by CAS number (ascending)
df_sorted = df.sort_values(by="cas")
print(df_sorted)
#         cas          name
# 1   50-00-0  Formaldehyde
# 3   67-56-1      Methanol
# 0   58-08-2      Caffeine
# 2  7732-18-5         Water

# Sort by CAS number (descending)
df_desc = df.sort_values(by="cas", ascending=False)
```

### Validation of CAS number structure

The CASDtype automatically validates CAS numbers when they are added to the array. Invalid CAS numbers will raise a `ValueError`:

```python
# This will raise a ValueError due to invalid checksum
try:
    df = pd.DataFrame({
        "cas": CASArray(["50-00-1"])  # Invalid checksum
    })
except ValueError as e:
    print(f"Validation error: {e}")
    # Validation error: Invalid CAS checksum for '50-00-1'

# Handle missing values with None or pd.NA
df = pd.DataFrame({
    "cas": CASArray(["50-00-0", None, "58-08-2"]),
    "name": ["Formaldehyde", "Unknown", "Caffeine"]
})

# Check for missing values
print(df["cas"].isna())
# 0    False
# 1     True
# 2    False
# Name: cas, dtype: bool
```

### Working with CAS columns

```python
# Create a Series with CAS numbers
cas_series = pd.Series(CASArray(["50-00-0", "58-08-2", "7732-18-5"]))

# Get min and max CAS numbers
print(cas_series.min())  # 50-00-0
print(cas_series.max())  # 7732-18-5

# Filter DataFrames
df = pd.DataFrame({
    "cas": CASArray(["50-00-0", "58-08-2", "7732-18-5"]),
    "toxicity": ["high", "medium", "low"]
})

high_toxicity = df[df["toxicity"] == "high"]
print(high_toxicity["cas"].iloc[0])  # 50-00-0

# Concatenate DataFrames with CAS columns
df1 = pd.DataFrame({"cas": CASArray(["50-00-0", "58-08-2"])})
df2 = pd.DataFrame({"cas": CASArray(["7732-18-5"])})
df_combined = pd.concat([df1, df2], ignore_index=True)
print(df_combined["cas"].dtype)  # CAS
```

### Joining DataFrames on CAS numbers

You can perform inner joins (or other merge operations) on DataFrames using CAS numbers as the key:

```python
# Create two DataFrames with CAS number columns
chemicals_df = pd.DataFrame({
    "cas": CASArray(["50-00-0", "58-08-2", "7732-18-5", "67-56-1"]),
    "name": ["Formaldehyde", "Caffeine", "Water", "Methanol"],
    "formula": ["CH2O", "C8H10N4O2", "H2O", "CH3OH"]
})

properties_df = pd.DataFrame({
    "cas": CASArray(["58-08-2", "7732-18-5", "64-17-5", "50-00-0"]),
    "boiling_point": [178, 100, 78, -19],
    "state": ["solid", "liquid", "liquid", "gas"]
})

# Perform inner join on CAS number
merged_df = pd.merge(chemicals_df, properties_df, on="cas", how="inner")

print(merged_df)
#        cas          name    formula  boiling_point   state
# 0  50-00-0  Formaldehyde       CH2O            -19     gas
# 1  58-08-2      Caffeine  C8H10N4O2            178   solid
# 2  7732-18-5       Water        H2O            100  liquid

# The CAS dtype is preserved after merge
print(merged_df["cas"].dtype)  # CAS

# You can also do left, right, or outer joins
left_join = pd.merge(chemicals_df, properties_df, on="cas", how="left")
# This will include Methanol with NaN values for boiling_point and state
```

### Converting to dictionary or JSON

```python
df = pd.DataFrame({
    "cas": CASArray(["50-00-0", "58-08-2"]),
    "name": ["Formaldehyde", "Caffeine"]
})

# Convert to dictionary (CAS objects preserved)
records = df.to_dict(orient="records")
print(records[0]["cas"])  # CAS(num='50-00-0')

# Convert CAS objects in dataframes to strings for JSON serialization as strings
df["cas_str"] = df["cas"].astype(str)
json_data = df.to_json(orient="records")
```

## Requirements

- Python 3.11+
- Pydantic 2.8.2+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
