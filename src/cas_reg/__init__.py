"""
Implementation of a pydantic BaseModel class for CAS Registry Numbers® (CAS RN®)*;
Provides validation on whether the provided number meets constraints imposed on format and values,
including verifying checkdigit character and length of the 3 fields.

See Chemical Abstract Services specification for more detail on CAS Registry Numbers®
https://www.cas.org/training/documentation/chemical-substances/checkdig

*note Chemical Abstracts is picky about including the registered trademark symbol when referring to their system
"""
from pydantic import BaseModel, Field, field_validator
from typing import ClassVar, Self
import re

class CAS(BaseModel):
    num: str = Field(..., frozen=True, serialization_alias='CAS',min_length=7, max_length=12)
    cas_regex: ClassVar[str] = r"^[1-9]{1}\d{1,6}-\d{2}-\d$"

    @field_validator("num")
    def validate_num(cls, v: str) -> str:
        if not re.match(cls.cas_regex, v):
            msg = f"CAS number {v} is not in the correct format (1X-XX-X up to 9XXXXXX-XX-X)"
            raise ValueError(msg)
        return v
    
    @classmethod
    def compute_checkdigit(cls, num: str):
        cas_reversed_list = map(int, num.replace("-", "")[::-1])
        return sum([c*i for (c, i) in enumerate(cas_reversed_list)]) % 10
    
    @field_validator("num")
    def verify_checkdigit(cls, v: str):
        if cls.compute_checkdigit(num=v) != int(v[-1]):
            msg: str = f"Invalid checkdigit character {int(v[-1])} for CAS number {v}"
            raise ValueError(msg)
        return v
    
    def __lt__(self, other: Self):
        # implements sorting based on number values rather than string sorting
        return int(self.num.replace("-", "")) < int(other.num.replace("-", ""))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CAS):
            return NotImplemented
        return self.num == other.num
    
    def __hash__(self):
        return hash(self.num)
    
    def __str__(self):
        return self.num


__all__ = ["CAS"]
