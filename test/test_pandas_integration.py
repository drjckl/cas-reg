"""Tests for pandas extension types for CAS numbers."""
import pytest

SKIP_PANDAS_TESTS = False # Make sure default is False modifying if import is successful

# Check if pandas is available
try:
    import pandas as pd
    import numpy as np
    from cas_reg import CAS
    from cas_reg.pandas_ext import CASDtype, CASArray

except ImportError:
    SKIP_PANDAS_TESTS = True
    pytestmark = pytest.mark.skip(reason="pandas not installed")
    

@pytest.mark.skipif(SKIP_PANDAS_TESTS, reason="pandas not available")
class TestCASDtype:
    """Test the CASDtype extension dtype."""

    def test_dtype_name(self):
        """Test that dtype has correct name."""
        dtype = CASDtype()
        assert dtype.name == "CAS"

    def test_dtype_type(self):
        """Test that dtype type is CAS."""
        dtype = CASDtype()
        assert dtype.type == CAS

    def test_construct_array_type(self):
        """Test that dtype constructs correct array type."""
        dtype = CASDtype()
        assert dtype.construct_array_type() == CASArray

    def test_na_value(self):
        """Test that NA value is pd.NA."""
        dtype = CASDtype()
        assert dtype.na_value is pd.NA

    def test_repr(self):
        """Test string representation."""
        dtype = CASDtype()
        assert repr(dtype) == "CAS"


@pytest.mark.skipif(SKIP_PANDAS_TESTS, reason="pandas not available")
class TestCASArray:
    """Test the CASArray extension array."""

    def test_create_from_strings(self):
        """Test creating CASArray from string list."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        assert len(arr) == 3
        assert isinstance(arr[0], CAS)
        assert str(arr[0]) == "50-00-0"

    def test_create_from_cas_objects(self):
        """Test creating CASArray from CAS objects."""
        cas_list = [CAS(num="50-00-0"), CAS(num="50-01-1")]
        arr = CASArray(cas_list)
        assert len(arr) == 2
        assert arr[0] == cas_list[0]

    def test_create_with_missing_values(self):
        """Test creating CASArray with NA values."""
        arr = CASArray(["50-00-0", None, "58-08-2"])
        assert len(arr) == 3
        assert arr[0] is not None
        assert arr[1] is None
        assert arr[2] is not None

    def test_dtype_property(self):
        """Test that array has correct dtype."""
        arr = CASArray(["50-00-0", "50-01-1"])
        assert isinstance(arr.dtype, CASDtype)

    def test_isna(self):
        """Test NA detection."""
        arr = CASArray(["50-00-0", None, "58-08-2"])
        na_mask = arr.isna()
        assert na_mask[0] == False
        assert na_mask[1] == True
        assert na_mask[2] == False

    def test_take(self):
        """Test take operation."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        result = arr.take([0, 2])
        assert len(result) == 2
        assert str(result[0]) == "50-00-0"
        assert str(result[1]) == "58-08-2"

    def test_take_with_fill(self):
        """Test take operation with fill value."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        result = arr.take([0, -1, 2], allow_fill=True, fill_value=None)
        assert len(result) == 3
        assert result[1] is None

    def test_copy(self):
        """Test array copy."""
        arr = CASArray(["50-00-0", "50-01-1"])
        arr_copy = arr.copy()
        assert len(arr_copy) == len(arr)
        assert arr_copy is not arr
        assert arr_copy[0] == arr[0]

    def test_concat(self):
        """Test concatenation of arrays."""
        arr1 = CASArray(["50-00-0", "50-01-1"])
        arr2 = CASArray(["58-08-2"])
        result = CASArray._concat_same_type([arr1, arr2])
        assert len(result) == 3
        assert str(result[0]) == "50-00-0"
        assert str(result[2]) == "58-08-2"

    def test_getitem_scalar(self):
        """Test scalar indexing."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        assert isinstance(arr[0], CAS)
        assert str(arr[1]) == "50-01-1"

    def test_getitem_slice(self):
        """Test slice indexing."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        result = arr[1:]
        assert isinstance(result, CASArray)
        assert len(result) == 2

    def test_setitem(self):
        """Test setting values."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        arr[1] = "58-08-2"
        assert str(arr[1]) == "58-08-2"

    def test_reduce_min(self):
        """Test min reduction."""
        arr = CASArray(["50-01-1", "50-00-0", "58-08-2"])
        result = arr._reduce("min")
        assert str(result) == "50-00-0"

    def test_reduce_max(self):
        """Test max reduction."""
        arr = CASArray(["50-01-1", "50-00-0", "58-08-2"])
        result = arr._reduce("max")
        assert str(result) == "58-08-2"

    def test_reduce_invalid_operation(self):
        """Test that invalid reduction raises error."""
        arr = CASArray(["50-00-0", "50-01-1"])
        with pytest.raises(TypeError):
            arr._reduce("sum")

    def test_repr(self):
        """Test string representation."""
        arr = CASArray(["50-00-0", "50-01-1"])
        repr_str = repr(arr)
        assert "CASArray" in repr_str
        assert "50-00-0" in repr_str

    def test_create_with_invalid_type_in_list(self):
        """Test creating CASArray with invalid type in list (line 67)."""
        # Non-string, non-CAS values should be converted to None
        arr = CASArray(["50-00-0", 123, "58-08-2"])
        assert len(arr) == 3
        assert arr[0] is not None
        assert arr[1] is None  # Integer converted to None
        assert arr[2] is not None

    def test_create_from_numpy_array_with_copy(self):
        """Test creating CASArray from numpy array with copy=True (line 71)."""
        original = np.array([CAS(num="50-00-0"), CAS(num="50-01-1")], dtype=object)
        arr = CASArray(original, copy=True)
        assert len(arr) == 2
        # Verify it's a copy, not a reference
        assert arr._data is not original

    def test_create_from_invalid_type_raises_error(self):
        """Test that creating CASArray from invalid type raises TypeError (line 75)."""
        with pytest.raises(TypeError, match="Cannot construct CASArray from"):
            CASArray("not a list or array")

    def test_from_factorized(self):
        """Test _from_factorized method (lines 87-88)."""
        values = np.array([CAS(num="50-00-0"), CAS(num="50-01-1")], dtype=object)
        original = CASArray(["50-00-0", "50-01-1"])
        arr = CASArray._from_factorized(values, original)
        assert len(arr) == 2
        assert isinstance(arr, CASArray)

    def test_eq_with_invalid_type(self):
        """Test __eq__ returns NotImplemented for invalid types (line 105)."""
        arr = CASArray(["50-00-0", "50-01-1"])
        result = arr.__eq__("not a CASArray")
        assert result is NotImplemented

    def test_eq_with_cas_array(self):
        """Test __eq__ with another CASArray (line 104)."""
        arr1 = CASArray(["50-00-0", "50-01-1"])
        arr2 = CASArray(["50-00-0", "50-01-1"])
        # This will compare the underlying numpy arrays
        result = arr1.__eq__(arr2)
        # The result should be a numpy array of booleans
        assert len(result) == 2

    def test_nbytes_property(self):
        """Test nbytes property (line 115)."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        nbytes = arr.nbytes
        assert isinstance(nbytes, int)
        assert nbytes > 0

    def test_repr_long_array_truncation(self):
        """Test that long arrays are truncated in repr (line 155)."""
        # Create array with more than 10 elements (all valid CAS numbers)
        arr = CASArray(["50-00-0", "50-01-1", "50-02-2", "50-03-3", "50-04-4",
                        "50-06-6", "50-07-7", "50-28-2", "50-29-3", "50-70-4",
                        "50-78-2", "50-99-7"])
        repr_str = repr(arr)
        assert "..." in repr_str  # Should show truncation
        assert "CASArray" in repr_str

    def test_setitem_with_na_value(self):
        """Test __setitem__ with NA value (lines 162-163)."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        arr[1] = pd.NA
        assert arr[1] is None

    def test_formatter_method(self):
        """Test _formatter method (lines 168-174)."""
        arr = CASArray(["50-00-0", None, "58-08-2"])
        formatter = arr._formatter()
        assert callable(formatter)
        # Test formatting a CAS object
        assert formatter(arr[0]) == "50-00-0"
        # Test formatting None
        assert formatter(None) == str(pd.NA)


@pytest.mark.skipif(SKIP_PANDAS_TESTS, reason="pandas not available")
class TestPandasIntegration:
    """Test integration with pandas DataFrames."""

    def test_create_series(self):
        """Test creating a pandas Series with CAS dtype."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        series = pd.Series(arr)
        assert len(series) == 3
        assert isinstance(series.dtype, CASDtype)

    def test_create_dataframe(self):
        """Test creating a DataFrame with CAS column."""
        df = pd.DataFrame({
            "cas": CASArray(["50-00-0", "50-01-1", "58-08-2"]),
            "name": ["Formaldehyde", "Sodium chloride", "Caffeine"]
        })
        assert len(df) == 3
        assert isinstance(df["cas"].dtype, CASDtype)

    def test_dataframe_operations(self):
        """Test basic DataFrame operations."""
        df = pd.DataFrame({
            "cas": CASArray(["50-00-0", "50-01-1", "58-08-2"]),
            "value": [1, 2, 3]
        })
        # Test filtering
        filtered = df[df["value"] > 1]
        assert len(filtered) == 2
        assert isinstance(filtered["cas"].dtype, CASDtype)

    def test_series_with_missing_values(self):
        """Test Series with missing values."""
        arr = CASArray(["50-00-0", None, "58-08-2"])
        series = pd.Series(arr)
        assert series.isna().sum() == 1

    def test_series_indexing(self):
        """Test indexing Series."""
        arr = CASArray(["50-00-0", "50-01-1", "58-08-2"])
        series = pd.Series(arr)
        assert isinstance(series.iloc[0], CAS)
        assert str(series.iloc[1]) == "50-01-1"

    def test_dataframe_concat(self):
        """Test concatenating DataFrames with CAS columns."""
        df1 = pd.DataFrame({"cas": CASArray(["50-00-0", "50-01-1"])})
        df2 = pd.DataFrame({"cas": CASArray(["58-08-2"])})
        result = pd.concat([df1, df2], ignore_index=True)
        assert len(result) == 3
        assert isinstance(result["cas"].dtype, CASDtype)

    def test_from_sequence(self):
        """Test creating array from sequence."""
        arr = CASArray._from_sequence(["50-00-0", "50-01-1", "58-08-2"])
        assert len(arr) == 3
        assert isinstance(arr, CASArray)

    def test_invalid_cas_number(self):
        """Test that invalid CAS numbers raise errors."""
        with pytest.raises(ValueError):
            CASArray(["50-00-1"])  # Invalid checksum

    def test_dataframe_to_dict(self):
        """Test converting DataFrame with CAS column to dict."""
        df = pd.DataFrame({
            "cas": CASArray(["50-00-0", "50-01-1"]),
            "value": [1, 2]
        })
        result = df.to_dict(orient="records")
        assert len(result) == 2
        # CAS objects should be serialized as CAS objects
        assert isinstance(result[0]["cas"], CAS)

    def test_cast_object_column_to_cas_dtype(self):
        """Test casting a DataFrame column from object dtype to CASDtype."""
        # Create DataFrame with CAS numbers as object dtype (strings)
        df = pd.DataFrame({
            "cas": ["50-00-0", "50-01-1", "58-08-2"],
            "name": ["Formaldehyde", "Sodium chloride", "Caffeine"]
        })
        # Verify initial dtype is object
        assert df["cas"].dtype == object

        # Cast the column to CASDtype
        df["cas"] = df["cas"].astype(CASDtype())

        # Verify the dtype is now CASDtype
        assert isinstance(df["cas"].dtype, CASDtype)
        # Verify values are CAS objects
        assert isinstance(df["cas"].iloc[0], CAS)
        assert str(df["cas"].iloc[0]) == "50-00-0"
        assert str(df["cas"].iloc[1]) == "50-01-1"
        assert str(df["cas"].iloc[2]) == "58-08-2"

    def test_sort_dataframe_by_cas_column(self):
        """Test sorting a DataFrame by a CASDtype column."""
        # Create DataFrame with CAS numbers in non-sorted order
        df = pd.DataFrame({
            "cas": CASArray(["58-08-2", "50-00-0", "50-01-1", "50-70-4"]),
            "name": ["Caffeine", "Formaldehyde", "Sodium chloride", "Pentachlorophenol"]
        })

        # Sort by CAS column
        df_sorted = df.sort_values(by="cas")

        # Verify the DataFrame is sorted correctly
        assert str(df_sorted["cas"].iloc[0]) == "50-00-0"
        assert str(df_sorted["cas"].iloc[1]) == "50-01-1"
        assert str(df_sorted["cas"].iloc[2]) == "50-70-4"
        assert str(df_sorted["cas"].iloc[3]) == "58-08-2"

        # Verify names correspond to sorted CAS numbers
        assert df_sorted["name"].iloc[0] == "Formaldehyde"
        assert df_sorted["name"].iloc[1] == "Sodium chloride"

        # Test descending sort
        df_sorted_desc = df.sort_values(by="cas", ascending=False)
        assert str(df_sorted_desc["cas"].iloc[0]) == "58-08-2"
        assert str(df_sorted_desc["cas"].iloc[3]) == "50-00-0"

    def test_merge_dataframes_on_cas_column(self):
        """Test merging DataFrames on CASDtype column."""
        # Create two DataFrames with CAS columns
        chemicals_df = pd.DataFrame({
            "cas": CASArray(["50-00-0", "58-08-2", "7732-18-5", "67-56-1"]),
            "name": ["Formaldehyde", "Caffeine", "Water", "Methanol"]
        })

        properties_df = pd.DataFrame({
            "cas": CASArray(["58-08-2", "7732-18-5", "64-17-5", "50-00-0"]),
            "boiling_point": [178, 100, 78, -19]
        })

        # Perform inner join on CAS number
        merged_df = pd.merge(chemicals_df, properties_df, on="cas", how="inner")

        # Verify merge results
        assert len(merged_df) == 3  # Only 3 CAS numbers in common
        assert isinstance(merged_df["cas"].dtype, CASDtype)

        # Verify the merged data
        cas_values = [str(cas) for cas in merged_df["cas"]]
        assert "50-00-0" in cas_values
        assert "58-08-2" in cas_values
        assert "7732-18-5" in cas_values
        assert "67-56-1" not in cas_values  # Not in properties_df
        assert "64-17-5" not in cas_values  # Not in chemicals_df

        # Verify values are still CAS objects
        assert isinstance(merged_df["cas"].iloc[0], CAS)

        # Test left join
        left_merged = pd.merge(chemicals_df, properties_df, on="cas", how="left")
        assert len(left_merged) == 4  # All chemicals included
        assert isinstance(left_merged["cas"].dtype, CASDtype)
