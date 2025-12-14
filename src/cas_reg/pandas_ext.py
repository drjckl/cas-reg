"""
Pandas extension types for CAS Registry Numbers.

This module provides custom pandas dtypes and arrays for working with CAS numbers
in pandas DataFrames and Series. It requires pandas and numpy to be installed.

To use these features, install cas-reg with the pandas extra:
    pip install cas-reg[pandas]
"""
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.core.dtypes.base import register_extension_dtype

from cas_reg import CAS


@register_extension_dtype
class CASDtype(ExtensionDtype):
    """Custom pandas dtype for CAS Registry Numbers."""

    name = "CAS"
    type = CAS
    _metadata = ("name",)

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return CASArray

    @property
    def na_value(self):
        """The missing value representation for this dtype."""
        return pd.NA

    def __repr__(self) -> str:
        return self.name


class CASArray(ExtensionArray):
    """Pandas ExtensionArray for CAS Registry Numbers."""

    def __init__(self, values, dtype=None, copy=False):
        """
        Initialize a CASArray.

        Parameters
        ----------
        values : sequence
            Values to store in the array. Can be CAS objects or strings.
        dtype : CASDtype, optional
            The dtype for this array.
        copy : bool, default False
            Whether to copy the input values.
        """
        if isinstance(values, list):
            # Convert strings to CAS objects
            cas_values = []
            for v in values:
                if pd.isna(v):
                    cas_values.append(None)
                elif isinstance(v, CAS):
                    cas_values.append(v if not copy else CAS(num=v.num))
                elif isinstance(v, str): # yes, we should accept strings in a list
                    cas_values.append(CAS(num=v))
                else:
                    cas_values.append(None) # test this
            self._data = np.array(cas_values, dtype=object)
        elif isinstance(values, np.ndarray):
            # Check if array contains strings that need conversion
            if len(values) > 0 and isinstance(values[0], str):
                # Convert strings to CAS objects
                cas_values = []
                for v in values:
                    if pd.isna(v):
                        cas_values.append(None)
                    elif isinstance(v, str):
                        cas_values.append(CAS(num=v))
                    else:
                        cas_values.append(None)
                self._data = np.array(cas_values, dtype=object)
            else:
                # Already contains CAS objects or is empty
                if copy:
                    self._data = values.copy() # test that we have copied rather than referenced
                else:
                    self._data = values
        else:
            raise TypeError(f"Cannot construct CASArray from {type(values)}") # test that this is raised properly

        self._dtype = dtype if dtype is not None else CASDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new CASArray from a sequence of scalars."""
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct a CASArray after factorization."""
        _ = original  # unused but required by interface
        return cls(values)

    def __getitem__(self, item):
        """Select a subset of self."""
        if isinstance(item, int):
            return self._data[item]
        else:
            return type(self)(self._data[item])

    def __len__(self) -> int:
        """Length of this array."""
        return len(self._data)

    def __eq__(self, other):
        """Check equality element-wise."""
        if isinstance(other, (CASArray, CAS)):
            return self._data == other
        return NotImplemented # test that this is raised

    @property
    def dtype(self) -> CASDtype:
        """The dtype for this array."""
        return self._dtype

    @property
    def nbytes(self) -> int:
        """The number of bytes needed to store this object in memory."""
        return self._data.nbytes # test this functionality

    def isna(self):
        """Boolean array indicating which values are missing."""
        return np.array([v is None or pd.isna(v) for v in self._data])

    def take(self, indices, allow_fill=False, fill_value=None):
        """Take elements from an array."""
        if allow_fill:
            result = [self._data[i] if i != -1 else fill_value for i in indices]
        else:
            result = [self._data[i] for i in indices]
        return type(self)(result, dtype=self.dtype)

    def copy(self):
        """Return a copy of the array."""
        return type(self)(self._data.copy(), dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple CASArray objects."""
        data = np.concatenate([arr._data for arr in to_concat])
        return cls(data)

    def _reduce(self, name, skipna=True, **kwargs):
        """Return a scalar result of performing the reduction operation."""
        _ = skipna, kwargs  # unused but required by interface
        if name == "min":
            valid = [v for v in self._data if v is not None]
            return min(valid) if valid else pd.NA
        elif name == "max":
            valid = [v for v in self._data if v is not None]
            return max(valid) if valid else pd.NA
        else:
            raise TypeError(f"Cannot perform reduction '{name}' with CAS dtype")

    def __repr__(self) -> str:
        """String representation of the array."""
        data_str = ", ".join([str(v) if v is not None else "NA" for v in self._data[:10]])
        if len(self._data) > 10:
            data_str += ", ..." # test that long arrays are truncated
        return f"CASArray([{data_str}], dtype=CAS)"  # is this normal for long arrays to truncate in repr

    def __setitem__(self, key, value):
        """Set values in the array."""
        if isinstance(value, str):
            value = CAS(num=value)
        elif pd.isna(value): # test this implemented
            value = None
        self._data[key] = value 
        # should we override with str so that setting item or items is successful?

    def _formatter(self, boxed=False):
        """Formatting function for pretty printing."""
        _ = boxed  # unused but required by interface
        def fmt(x):
            if x is None or pd.isna(x):
                return str(pd.NA)
            return str(x)
        return fmt  # test pretty print function


__all__ = ["CASDtype", "CASArray"]
