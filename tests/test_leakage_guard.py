import unittest

from src.validation import validate_evaluation_cities, validate_feature_columns, validate_test_cities


class FeatureLeakageTests(unittest.TestCase):
    def test_distinct_features_are_accepted(self):
        validate_feature_columns(["DNI", "T"], "GI")

    def test_target_in_features_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Target column"):
            validate_feature_columns(["DNI", "GI"], "GI")

    def test_duplicate_or_empty_features_are_rejected(self):
        for features in ([], ["DNI", "DNI"]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                validate_feature_columns(features, "GI")

    def test_evaluation_must_use_cities_held_out_during_training(self):
        validate_evaluation_cities(["Tunis", "Sfax"], ["Tunis"])
        with self.assertRaisesRegex(ValueError, "not held out"):
            validate_evaluation_cities(["Tunis"], ["Sfax"])

    def test_invalid_test_city_groups_are_rejected(self):
        for cities in ([], "Tunis", ["Tunis", "Tunis"]):
            with self.subTest(cities=cities), self.assertRaises(ValueError):
                validate_test_cities(cities)


if __name__ == "__main__":
    unittest.main()
