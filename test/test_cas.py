import pytest
import json
from typing import Literal

from cas_reg import CAS

cas_obj1_str: Literal["50-00-0"] = "50-00-0"
cas_obj2_str: Literal["50-01-1"] = "50-01-1"
cas_caffine_str: Literal["58-08-2"] = "58-08-2"

def test_cas_creation_positional():
    with pytest.raises(TypeError):
        CAS("50-00-0")

def test_cas_number_creation():
    CAS(num="50-00-0")
    CAS(num="50-01-1")
    CAS(num="58-08-2")
    
def test_cas_number_bad_format():
    with pytest.raises(ValueError):
        CAS(num="10000000-00-1")
    with pytest.raises(ValueError):
        CAS(num="50-001-1")
    with pytest.raises(ValueError):
        CAS(num="50-00-01")
    
def test_import_cas_bad_checksum():
    with pytest.raises(ValueError):
        CAS(num="50-00-1")
        
def test_cas_string_representation():
    assert str(CAS(num="50-00-0")) == "50-00-0"
        
def test_cas_number_comparison():
    assert CAS(num=cas_obj1_str) < CAS(num=cas_obj2_str)
    
def test_is():
    a: CAS = CAS(num="50-01-1")
    b: CAS = CAS(num="50-01-1")
    assert (a is not b)
    
def test_cas_number_equality():
    assert CAS(num="50-01-1") == CAS(num="50-01-1")
    
def test_cas_number_inequality_to_non_cas_number():
    assert CAS(num="50-00-0") != "50-00-0"

def test_cas_eq_returns_not_implemented_for_non_cas():
    """Test that __eq__ returns NotImplemented for non-CAS objects (line 44)."""
    cas = CAS(num="50-00-0")
    result = cas.__eq__("50-00-0")
    assert result is NotImplemented
    # Also test with other types
    assert cas.__eq__(123) is NotImplemented
    assert cas.__eq__(None) is NotImplemented
        
def test_cas_to_string():
    assert str(CAS(num=cas_obj1_str)) == "50-00-0"
    
def test_cas_list_to_string():
    test_list: list[CAS] = [CAS(num=cas_obj1_str), CAS(num=cas_obj2_str)]
    string_list: list[str] = [str(cas) for cas in test_list]
    assert string_list == ['50-00-0', '50-01-1']
    
def test_cas_regexp():
    assert CAS.cas_regex == r"^[1-9]{1}\d{1,6}-\d{2}-\d$"
    
def test_cas_checksum_function():
    assert CAS.compute_checkdigit(num="50-00-0") == 0
    
def test_sort_cas_numbers():
    cas_obj1: CAS = CAS(num=cas_obj1_str)
    cas_obj2: CAS = CAS(num=cas_obj2_str)
    cas_obj3: CAS = CAS(num=cas_caffine_str)
    assert sorted([cas_obj2, cas_obj1, cas_obj3]) == [cas_obj1, cas_obj2, cas_obj3]
    print(sorted([cas_obj2, cas_obj1, cas_obj3]))
    
    
def test_model_dump():
    model_dump_output: dict = CAS(num=cas_obj1_str).model_dump(by_alias=True)
    assert {'CAS': '50-00-0'} == model_dump_output


def test_model_dump_json():
    json_output: str = CAS(num=cas_obj1_str).model_dump_json(by_alias=True)
    assert isinstance(json_output, str)
    assert {'CAS': '50-00-0'} == json.loads(json_output)
    
def test_set_equality():
    list1: list[CAS] = [CAS(num="50-00-0"), CAS(num="50-01-1"), CAS(num="58-08-2"), CAS(num="50-00-0")]
    list2: list[CAS] = [CAS(num="50-00-0"), CAS(num="50-01-1"), CAS(num="58-08-2")]
    assert set(list1) == set(list2)
    
