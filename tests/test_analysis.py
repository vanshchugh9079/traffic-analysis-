"""
Unit tests for AccidentAnalyzer module.
Verifies statistical computations, grouping, aggregations, and edge cases.
"""
import os
import sys
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
    from modules.analysis import AccidentAnalyzer
except ImportError:
    from TrafficAccidentAnalyzer.modules.analysis import AccidentAnalyzer


class TestAccidentAnalyzer(unittest.TestCase):
    """Test suite for accident data analysis operations."""

    def setUp(self):
        """Creates a deterministic test dataset for analysis."""
        self.sample_data = pd.DataFrame({
            "Accident_ID": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "Severity": ["Fatal", "Serious", "Minor", "Minor", "Fatal", "Minor", "Serious", "Minor"],
            "City": ["Chicago", "New York", "Chicago", "Seattle", "Chicago", "New York", "Seattle", "Houston"],
            "Hour": [8, 18, 8, 14, 8, 22, 18, 8],
            "Month": ["Jan", "Jan", "Feb", "Feb", "Mar", "Jan", "Feb", "Mar"],
            "Weekday": ["Monday", "Monday", "Tuesday", "Wednesday", "Friday", "Saturday", "Sunday", "Monday"],
            "Weather": ["Clear", "Rainy", "Clear", "Snow", "Foggy", "Clear", "Rainy", "Clear"],
            "Road_Surface": ["Dry", "Wet", "Dry", "Icy", "Damp", "Dry", "Wet", "Dry"],
            "Vehicles_Involved": [2, 1, 3, 2, 4, 1, 2, 2],
            "Casualties": [1, 2, 0, 1, 3, 0, 1, 0],
            "Latitude": [41.87, 40.71, 41.88, 47.60, 41.85, 40.73, 47.61, 29.76],
            "Longitude": [-87.62, -74.00, -87.63, -122.33, -87.65, -74.02, -122.34, -95.36]
        })
        self.analyzer = AccidentAnalyzer(self.sample_data)

    def test_total_accident_counts(self):
        """Case 4: Calculating total accident counts."""
        kpis = self.analyzer.get_kpis()
        self.assertEqual(kpis["total_accidents"], 8)
        self.assertEqual(kpis["fatal_accidents"], 2)
        self.assertEqual(kpis["serious_accidents"], 2)
        self.assertEqual(kpis["minor_accidents"], 4)

    def test_severity_counts_and_percentages(self):
        """Case 5: Calculating severity counts and percentages."""
        sev_res = self.analyzer.analyze_severity()
        counts = sev_res["counts"]
        percentages = sev_res["percentages"]

        self.assertEqual(counts["Fatal"], 2)
        self.assertEqual(counts["Serious"], 2)
        self.assertEqual(counts["Minor"], 4)

        # 4 out of 8 is 50.0%, 2 out of 8 is 25.0%
        self.assertAlmostEqual(percentages["Minor"], 50.0, places=1)
        self.assertAlmostEqual(percentages["Fatal"], 25.0, places=1)
        self.assertAlmostEqual(percentages["Serious"], 25.0, places=1)
        self.assertEqual(sev_res["mode"], "Minor")

    def test_hourly_and_monthly_grouping(self):
        """Case 6: Grouping accident counts by hour and month."""
        time_res = self.analyzer.analyze_time()

        # Hours: 8 has four incidents (A1, A3, A5, A8)
        hourly = time_res["hourly_counts"]
        self.assertEqual(hourly[8], 4)
        self.assertEqual(hourly[18], 2)
        self.assertEqual(hourly[14], 1)
        self.assertEqual(time_res["peak_hour"], 8)
        self.assertEqual(time_res["peak_hour_count"], 4)

        # Months: Jan has 3, Feb has 3, Mar has 2
        monthly = time_res["monthly_counts"]
        self.assertEqual(monthly["Jan"], 3)
        self.assertEqual(monthly["Feb"], 3)
        self.assertEqual(monthly["Mar"], 2)

    def test_location_ranking(self):
        """Location grouping and Top-K extraction."""
        loc_res = self.analyzer.analyze_locations(top_n=3)
        top_locs = loc_res["top_locations"]

        # Chicago has 3 accidents, New York has 2, Seattle has 2
        self.assertEqual(top_locs["Chicago"], 3)
        self.assertEqual(loc_res["unique_count"], 4)
        self.assertTrue(loc_res["has_coordinates"])
        self.assertEqual(loc_res["valid_coordinate_count"], 8)

    def test_numerical_statistics_numpy(self):
        """NumPy statistical calculations on Vehicles_Involved and Casualties."""
        stats = self.analyzer.compute_statistics()
        self.assertIn("Vehicles_Involved", stats)
        veh_stats = stats["Vehicles_Involved"]

        # [2, 1, 3, 2, 4, 1, 2, 2] -> Mean = 17 / 8 = 2.125
        self.assertAlmostEqual(veh_stats["Mean"], 2.125, places=2)
        self.assertEqual(veh_stats["Median"], 2.0)
        self.assertEqual(veh_stats["Min"], 1.0)
        self.assertEqual(veh_stats["Max"], 4.0)

    def test_correlation_matrix(self):
        """Pearson correlation computation."""
        corr, ok = self.analyzer.compute_correlation()
        self.assertTrue(ok)
        self.assertIn("Vehicles_Involved", corr.columns)
        self.assertIn("Casualties", corr.columns)
        # Diagonal must be 1.0
        self.assertAlmostEqual(corr.loc["Vehicles_Involved", "Vehicles_Involved"], 1.0)

    def test_empty_dataset_handling(self):
        """Case 9: Handling an empty dataset gracefully without crash."""
        empty_analyzer = AccidentAnalyzer(pd.DataFrame())
        kpis = empty_analyzer.get_kpis()
        self.assertEqual(kpis["total_accidents"], 0)
        self.assertEqual(kpis["fatal_accidents"], 0)
        self.assertEqual(kpis["peak_hour"], "N/A")

        sev = empty_analyzer.analyze_severity()
        self.assertEqual(sev["total"], 0)

        t = empty_analyzer.analyze_time()
        self.assertEqual(t["hourly_counts"].sum(), 0)

        stats = empty_analyzer.compute_statistics()
        self.assertEqual(len(stats), 0)

        corr, ok = empty_analyzer.compute_correlation()
        self.assertFalse(ok)
        self.assertTrue(corr.empty)

    def test_missing_optional_columns(self):
        """Case 8: Handling missing optional columns (e.g. without Lat/Long)."""
        minimal_data = pd.DataFrame({
            "Accident_ID": ["A1", "A2"],
            "Severity": ["Fatal", "Minor"]
        })
        min_analyzer = AccidentAnalyzer(minimal_data)
        kpis = min_analyzer.get_kpis()
        self.assertEqual(kpis["total_accidents"], 2)
        self.assertEqual(kpis["fatal_accidents"], 1)

        loc = min_analyzer.analyze_locations()
        self.assertFalse(loc["has_coordinates"])
        self.assertEqual(loc["valid_coordinate_count"], 0)

        time_res = min_analyzer.analyze_time()
        self.assertEqual(time_res["hourly_counts"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
