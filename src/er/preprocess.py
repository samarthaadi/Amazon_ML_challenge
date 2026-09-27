"""Stage 1: retain originals and derive representations.

Implementation TBD: normalization, learned representations, and their fitting.
legacy_v1 is an adapter for the existing comparison baseline, not a new choice.
"""
from .data import legacy


def fit(training_records, config):
    if config["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: preprocessing")
    return {"method": "legacy_v1", "learned": False}


def transform(records, state):
    if state["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: preprocessing")
    result = records.copy()
    baseline = legacy()
    result["name_normalized"] = result.business_name.map(baseline.clean)
    result["address_normalized"] = result.business_address.map(baseline.clean)
    result["name_expanded"] = result.business_name.map(baseline.expanded)
    return result


LEXICAL_VERSION = "legacy-expanded-name-clean-address-v1"
DENSE_VERSION = "original-fields-whitespace-v1"


def blocking_records(records):
    """Retain originals; lexical cleanup and dense serialization are separate."""
    import pandas as pd
    result = records.copy()
    def text(value):
        return "" if value is None or pd.isna(value) else " ".join(str(value).split())
    names, addresses = result.business_name.map(text), result.business_address.map(text)
    result["lexical_name"] = names.map(legacy().expanded)
    result["lexical_address"] = addresses.map(legacy().clean)
    result["dense_text"] = [" ".join(p for p in (f"Name: {n}" if n else "", f"Address: {a}" if a else "") if p)
                            for n, a in zip(names, addresses)]
    result["missing_name"], result["missing_address"] = names.eq(""), addresses.eq("")
    return result
