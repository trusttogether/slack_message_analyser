import json
from typing import Dict, List, Any

def compare_json_structures(obj1: Any, obj2: Any, path: str = "") -> List[str]:
    """
    Compare two JSON objects and return differences
    """
    differences = []
    
    if type(obj1) != type(obj2):
        differences.append(f"Type mismatch at {path}: {type(obj1)} vs {type(obj2)}")
        return differences
    
    if isinstance(obj1, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())
        
        # Check for missing keys
        for key in keys1 - keys2:
            differences.append(f"Missing key at {path}.{key}")
        
        for key in keys2 - keys1:
            differences.append(f"Extra key at {path}.{key}")
        
        # Compare common keys
        for key in keys1 & keys2:
            differences.extend(compare_json_structures(obj1[key], obj2[key], f"{path}.{key}"))
    
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            differences.append(f"List length mismatch at {path}: {len(obj1)} vs {len(obj2)}")
        else:
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                differences.extend(compare_json_structures(item1, item2, f"{path}[{i}]"))
    
    elif obj1 != obj2:
        differences.append(f"Value mismatch at {path}: {obj1} vs {obj2}")
    
    return differences

def get_json_similarity(obj1: Any, obj2: Any) -> float:
    """
    Calculate similarity between two JSON objects (0-1)
    """
    differences = compare_json_structures(obj1, obj2)
    
    if not differences:
        return 1.0
    
    # Simple similarity based on number of differences
    # This is a basic implementation - could be enhanced
    max_differences = 100  # Arbitrary threshold
    similarity = max(0, 1 - len(differences) / max_differences)
    
    return similarity

def format_json_diff(differences: List[str]) -> str:
    """
    Format differences for human-readable output
    """
    if not differences:
        return "No differences found"
    
    return "\n".join([f"- {diff}" for diff in differences])
