from __future__ import annotations


def validate_feature_columns(features, target):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("The target column must be a non-empty string")
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("Feature columns must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("Feature names must be non-empty strings")
    if target in features:
        raise ValueError("Target column cannot also be an input feature")
    if len(set(features)) != len(features):
        raise ValueError("Feature columns must not contain duplicates")


def validate_test_cities(cities):
    if not isinstance(cities, (list, tuple)) or not cities:
        raise ValueError("test_cities must be a non-empty list")
    if any(not isinstance(city, str) or not city.strip() for city in cities):
        raise ValueError("test_cities must contain non-empty strings")
    if len(set(cities)) != len(cities):
        raise ValueError("test_cities must not contain duplicates")


def validate_evaluation_cities(training_holdout, evaluation_cities):
    validate_test_cities(training_holdout)
    validate_test_cities(evaluation_cities)
    unseen = set(evaluation_cities) - set(training_holdout)
    if unseen:
        raise ValueError(
            "Evaluation cities were not held out when the checkpoint was trained: "
            f"{sorted(unseen)}"
        )
