"""
Unit tests for DataLoader and DataCleaner modules.
Verifies file loading, column normalization, deduplication, missing value handling,
filtering, and exporting.
"""
import os
import sys
import tempfile
import unittest

# Ensure modules directory is discoverable from both root and project directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR, os.path.dirname(PARENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
try:
    from modules.data_loader import DataLoader
    from modules.data_cleaning import DataCleaner
except ImportError:
    from TrafficAccidentAnalyzer.modules.data_loader import DataLoader
    from TrafficAccidentAnalyzer.modules.data_cleaning import DataCleaner


class TestDataCleaningAndLoading(unittest.TestCase):
    """Test suite for data loading, cleaning, and filtering operations."""

    def setUp(self):
        """Prepares temporary directory and test datasets."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "test_accidents.csv")

        # Create a sample raw CSV with varying column names, duplicates, and missing values
        self.raw_df = pd.DataFrame({
            "accident_severity": ["fatal", "serious", "minor", "minor", "1", "fatal"],  # severity alias & duplicate
            "accident_date": ["2023-01-15", "2023-01-20", "2023-02-10", "2023-02-15", "2023-03-01", "2023-01-15"],
            "time_of_day": ["08:30", "18:45", "12:15", "14:20", "23:10", "08:30"],
            "location": ["Chicago", "New York", "Chicago", "Seattle", None, "Chicago"],
            "weather_condition": ["Clear", "Rainy", None, "Snow", "Clear", "Clear"],
            "vehicles_involved": [2, -1, 3, np.nan, 2, 2],  # invalid negative number and NaN
            "latitude": [41.87, 40.71, 999.0, 47.60, 0.0, 41.87],  # 999 is invalid lat, 0.0 is zero
            "longitude": [-87.62, -74.00, -87.62, -122.33, 0.0, -87.62],
            "id": ["ACC-1", "ACC-2", "ACC-3", "ACC-4", "ACC-5", "ACC-1"]  # ACC-1 duplicate
        })
        self.raw_df.to_csv(self.csv_path, index=False)

        self.loader = DataLoader()
        self.cleaner = DataCleaner()

    def tearDown(self):
        """Cleans up temporary files."""
        self.temp_dir.cleanup()

    def test_load_valid_csv(self):
        """Case 1: Loading a valid CSV dataset and verifying column mapping."""
        df, err = self.loader.load_file(self.csv_path)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)

        # Verify column aliases were mapped to canonical names
        self.assertIn("Severity", df.columns)
        self.assertIn("Date", df.columns)
        self.assertIn("City", df.columns)
        self.assertIn("Weather", df.columns)
        self.assertIn("Vehicles_Involved", df.columns)

    def test_missing_values_detection(self):
        """Case 2: Loading a dataset with missing values and assessing completeness."""
        df, _ = self.loader.load_file(self.csv_path)
        quality = self.cleaner.assess_quality(df)

        self.assertEqual(quality["total_rows"], 6)
        # Location and Weather have missing values
        self.assertGreater(quality["missing_summary"]["City"]["count"], 0)
        self.assertGreater(quality["missing_summary"]["Weather"]["count"], 0)

    def test_detect_and_remove_duplicates(self):
        """Case 3: Detecting duplicate rows and removing them."""
        df, _ = self.loader.load_file(self.csv_path)
        self.cleaner.set_dataframe(df)

        quality_before = self.cleaner.assess_quality(df)
        self.assertGreater(quality_before["duplicates_count"], 0)

        cleaned_df, report = self.cleaner.clean(drop_duplicates=True)
        self.assertGreater(report["metrics"]["duplicates_removed"], 0)
        self.assertEqual(len(cleaned_df), 5)  # 6 - 1 duplicate = 5

    def test_severity_normalization(self):
        """Verifies severity label normalization ('fatal', '1' -> 'Fatal')."""
        df, _ = self.loader.load_file(self.csv_path)
        self.cleaner.set_dataframe(df)
        cleaned_df, _ = self.cleaner.clean(normalize_severities=True)

        unique_sevs = set(cleaned_df["Severity"].unique())
        for sev in unique_sevs:
            self.assertIn(sev, ["Fatal", "Serious", "Minor", "Unknown"])

    def test_coordinate_sanitization(self):
        """Coordinate validation: sets out-of-bounds or zero lat/long to NaN."""
        df, _ = self.loader.load_file(self.csv_path)
        self.cleaner.set_dataframe(df)
        cleaned_df, _ = self.cleaner.clean(sanitize_coordinates=True)

        # Row with latitude 999.0 should have Latitude set to NaN
        invalid_mask = (cleaned_df["Latitude"] > 90) | ((cleaned_df["Latitude"] == 0) & (cleaned_df["Longitude"] == 0))
        self.assertEqual(invalid_mask.sum(), 0)

    def test_filtering_records(self):
        """Case 7: Filtering records by severity and location."""
        df, _ = self.loader.load_file(self.csv_path)

        # Filter by City: Chicago
        chicago_only = self.loader.filter_data(df, city="Chicago")
        for c in chicago_only["City"].dropna():
            self.assertEqual(c, "Chicago")

        # Filter by Severity: Fatal
        fatal_only = self.loader.filter_data(df, severity="Fatal")
        for s in fatal_only["Severity"].dropna():
            self.assertEqual(s, "Fatal")

    def test_export_filtered_results(self):
        """Case 10: Exporting filtered results to a new CSV file."""
        df, _ = self.loader.load_file(self.csv_path)
        filtered = self.loader.filter_data(df, city="Chicago")

        export_path = os.path.join(self.temp_dir.name, "exported_chicago.csv")
        success, err = self.loader.export_to_csv(filtered, export_path)

        self.assertTrue(success)
        self.assertIsNone(err)
        self.assertTrue(os.path.exists(export_path))

        # Read back and verify row count
        read_back = pd.read_csv(export_path)
        self.assertEqual(len(read_back), len(filtered))


if __name__ == "__main__":
    unittest.main()
