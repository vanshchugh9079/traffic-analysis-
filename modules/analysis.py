"""
Accident Analysis Module for Traffic Accident Analyzer.
Performs statistical computing, grouping, aggregations, cross-tabulations,
and metric derivations using Pandas and NumPy.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class AccidentAnalyzer:
    """Core analytical engine for processing accident dataset records."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df: pd.DataFrame = df.copy() if df is not None else pd.DataFrame()

    def set_data(self, df: pd.DataFrame):
        """Update working dataset."""
        self.df = df.copy()

    # -------------------------------------------------------------
    # 1. Summary Cards / KPIs
    # -------------------------------------------------------------
    def get_kpis(self) -> Dict[str, Any]:
        """
        Calculates high-level dashboard KPI metrics from the current dataset.
        Returns total accidents, fatal, serious, minor, peak hour, and unique locations.
        """
        if self.df.empty:
            return {
                "total_accidents": 0,
                "fatal_accidents": 0,
                "fatal_pct": 0.0,
                "serious_accidents": 0,
                "serious_pct": 0.0,
                "minor_accidents": 0,
                "minor_pct": 0.0,
                "peak_hour": "N/A",
                "peak_hour_count": 0,
                "unique_locations": 0,
                "avg_vehicles": 0.0
            }

        total = len(self.df)

        # Severity counts
        fatal = 0
        serious = 0
        minor = 0

        if "Severity" in self.df.columns:
            sev_counts = self.df["Severity"].astype(str).str.title().value_counts()
            fatal = int(sev_counts.get("Fatal", 0))
            serious = int(sev_counts.get("Serious", 0))
            minor = int(sev_counts.get("Minor", 0))

        # Peak Hour
        peak_hour_str = "N/A"
        peak_count = 0
        if "Hour" in self.df.columns and not self.df["Hour"].dropna().empty:
            hour_counts = self.df["Hour"].dropna().astype(int).value_counts()
            if not hour_counts.empty:
                peak_h = int(hour_counts.index[0])
                peak_count = int(hour_counts.iloc[0])
                # Format friendly 12-hour string (e.g., "18:00 (6 PM)")
                ampm = "AM" if peak_h < 12 else "PM"
                disp_h = peak_h if 1 <= peak_h <= 12 else (12 if peak_h == 0 else peak_h - 12)
                peak_hour_str = f"{peak_h:02d}:00 ({disp_h} {ampm})"

        # Unique locations
        unique_locs = 0
        if "City" in self.df.columns:
            unique_locs = int(self.df["City"].dropna().nunique())

        # Average vehicles
        avg_vehicles = 0.0
        if "Vehicles_Involved" in self.df.columns and not self.df["Vehicles_Involved"].dropna().empty:
            avg_vehicles = round(float(np.nanmean(self.df["Vehicles_Involved"].values)), 2)

        return {
            "total_accidents": total,
            "fatal_accidents": fatal,
            "fatal_pct": round(fatal / total * 100, 1) if total > 0 else 0.0,
            "serious_accidents": serious,
            "serious_pct": round(serious / total * 100, 1) if total > 0 else 0.0,
            "minor_accidents": minor,
            "minor_pct": round(minor / total * 100, 1) if total > 0 else 0.0,
            "peak_hour": peak_hour_str,
            "peak_hour_count": peak_count,
            "unique_locations": unique_locs,
            "avg_vehicles": avg_vehicles
        }

    # -------------------------------------------------------------
    # 2. Severity Analysis
    # -------------------------------------------------------------
    def analyze_severity(self) -> Dict[str, Any]:
        """
        Analyzes accident severity breakdown with counts, percentages,
        and identifies the dominant severity tier.
        """
        if self.df.empty or "Severity" not in self.df.columns:
            return {"counts": pd.Series(dtype=int), "percentages": pd.Series(dtype=float), "mode": "N/A", "total": 0}

        counts = self.df["Severity"].astype(str).str.title().value_counts()
        total = counts.sum()
        pcts = (counts / total * 100).round(2)
        mode_sev = str(counts.index[0]) if not counts.empty else "N/A"

        # Ensure canonical ordering if present
        order = ["Fatal", "Serious", "Minor"]
        ordered_counts = pd.Series(dtype=int)
        ordered_pcts = pd.Series(dtype=float)
        for cat in order:
            if cat in counts:
                ordered_counts[cat] = counts[cat]
                ordered_pcts[cat] = pcts[cat]
        # Include any other category
        for cat in counts.index:
            if cat not in order:
                ordered_counts[cat] = counts[cat]
                ordered_pcts[cat] = pcts[cat]

        return {
            "counts": ordered_counts,
            "percentages": ordered_pcts,
            "mode": mode_sev,
            "total": total
        }

    # -------------------------------------------------------------
    # 3. Time-based Analysis
    # -------------------------------------------------------------
    def analyze_time(self) -> Dict[str, Any]:
        """
        Computes accident occurrences grouped by Hour of Day, Day of Week,
        and Month of Year. Identifies peak periods.
        """
        result = {
            "hourly_counts": pd.Series(0, index=range(24)),
            "peak_hour": None,
            "peak_hour_count": 0,
            "weekday_counts": pd.Series(dtype=int),
            "peak_weekday": None,
            "monthly_counts": pd.Series(dtype=int),
            "peak_month": None,
            "peak_month_count": 0
        }

        if self.df.empty:
            return result

        # Hourly Analysis
        if "Hour" in self.df.columns:
            valid_hours = self.df["Hour"].dropna().astype(int)
            valid_hours = valid_hours[(valid_hours >= 0) & (valid_hours <= 23)]
            counts_h = valid_hours.value_counts()
            hourly = pd.Series(0, index=range(24))
            for h in range(24):
                if h in counts_h:
                    hourly[h] = int(counts_h[h])
            result["hourly_counts"] = hourly
            if not counts_h.empty:
                result["peak_hour"] = int(counts_h.index[0])
                result["peak_hour_count"] = int(counts_h.iloc[0])

        # Weekday Analysis
        weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if "Weekday" in self.df.columns:
            wk_counts = self.df["Weekday"].dropna().astype(str).value_counts()
            ordered_wk = pd.Series(0, index=weekdays_order)
            for w in weekdays_order:
                if w in wk_counts:
                    ordered_wk[w] = int(wk_counts[w])
            result["weekday_counts"] = ordered_wk
            if not wk_counts.empty:
                result["peak_weekday"] = str(wk_counts.index[0])

        # Monthly Analysis
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if "Month" in self.df.columns:
            mo_counts = self.df["Month"].dropna().astype(str).value_counts()
            ordered_mo = pd.Series(0, index=months_order)
            for m in months_order:
                if m in mo_counts:
                    ordered_mo[m] = int(mo_counts[m])
            result["monthly_counts"] = ordered_mo
            if not mo_counts.empty:
                result["peak_month"] = str(mo_counts.index[0])
                result["peak_month_count"] = int(mo_counts.iloc[0])

        return result

    # -------------------------------------------------------------
    # 4. Location Analysis
    # -------------------------------------------------------------
    def analyze_locations(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Groups accident counts by city/location, identifies the Top-N hotspots,
        and provides spatial bounding summary if coordinates are present.
        """
        if self.df.empty or "City" not in self.df.columns:
            return {
                "top_locations": pd.Series(dtype=int),
                "all_locations": pd.Series(dtype=int),
                "unique_count": 0,
                "has_coordinates": False,
                "valid_coordinate_count": 0
            }

        city_counts = self.df["City"].dropna().astype(str).str.strip().value_counts()
        top_locs = city_counts.head(top_n)

        # Check coordinates
        has_coords = False
        valid_coords_cnt = 0
        if "Latitude" in self.df.columns and "Longitude" in self.df.columns:
            lat = pd.to_numeric(self.df["Latitude"], errors="coerce")
            lon = pd.to_numeric(self.df["Longitude"], errors="coerce")
            valid_mask = lat.notna() & lon.notna() & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180) & ~((lat == 0) & (lon == 0))
            valid_coords_cnt = int(valid_mask.sum())
            has_coords = valid_coords_cnt > 0

        return {
            "top_locations": top_locs,
            "all_locations": city_counts,
            "unique_count": len(city_counts),
            "has_coordinates": has_coords,
            "valid_coordinate_count": valid_coords_cnt
        }

    # -------------------------------------------------------------
    # 5. Weather Analysis
    # -------------------------------------------------------------
    def analyze_weather(self) -> Dict[str, Any]:
        """
        Analyzes accident distribution across weather conditions and cross-tabulates
        with severity tiers.
        """
        if self.df.empty or "Weather" not in self.df.columns:
            return {"weather_counts": pd.Series(dtype=int), "crosstab": pd.DataFrame()}

        counts = self.df["Weather"].dropna().astype(str).str.strip().str.title().value_counts()

        crosstab = pd.DataFrame()
        if "Severity" in self.df.columns:
            crosstab = pd.crosstab(
                self.df["Weather"].dropna().astype(str).str.strip().str.title(),
                self.df["Severity"].dropna().astype(str).str.strip().str.title()
            )

        return {
            "weather_counts": counts,
            "crosstab": crosstab
        }

    # -------------------------------------------------------------
    # 6. Road and Lighting Analysis
    # -------------------------------------------------------------
    def analyze_road_and_light(self) -> Dict[str, Any]:
        """
        Analyzes road surface conditions and ambient lighting environment.
        """
        road_counts = pd.Series(dtype=int)
        light_counts = pd.Series(dtype=int)
        road_severity_ct = pd.DataFrame()

        if not self.df.empty:
            if "Road_Surface" in self.df.columns:
                road_counts = self.df["Road_Surface"].dropna().astype(str).str.strip().str.title().value_counts()
                if "Severity" in self.df.columns:
                    road_severity_ct = pd.crosstab(
                        self.df["Road_Surface"].dropna().astype(str).str.strip().str.title(),
                        self.df["Severity"].dropna().astype(str).str.strip().str.title()
                    )

            if "Light_Condition" in self.df.columns:
                light_counts = self.df["Light_Condition"].dropna().astype(str).str.strip().value_counts()

        return {
            "road_surface_counts": road_counts,
            "light_condition_counts": light_counts,
            "road_severity_crosstab": road_severity_ct
        }

    # -------------------------------------------------------------
    # 7. Vehicle Analysis
    # -------------------------------------------------------------
    def analyze_vehicles(self) -> Dict[str, Any]:
        """
        Computes distribution of vehicles involved in accidents.
        """
        if self.df.empty or "Vehicles_Involved" not in self.df.columns:
            return {
                "distribution": pd.Series(dtype=int),
                "mean": 0.0,
                "median": 0.0,
                "max": 0,
                "min": 0,
                "std": 0.0
            }

        nums = pd.to_numeric(self.df["Vehicles_Involved"], errors="coerce").dropna()
        nums = nums[nums > 0]

        if nums.empty:
            return {"distribution": pd.Series(dtype=int), "mean": 0.0, "median": 0.0, "max": 0, "min": 0, "std": 0.0}

        dist = nums.astype(int).value_counts().sort_index()
        arr = nums.to_numpy()

        return {
            "distribution": dist,
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "max": int(np.max(arr)),
            "min": int(np.min(arr)),
            "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0
        }

    # -------------------------------------------------------------
    # 8. Statistical Calculations (NumPy-Powered)
    # -------------------------------------------------------------
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates descriptive summary statistics using NumPy (Mean, Median,
        Std Dev, Variance, Min, Max, 25th, 50th, 75th, 90th percentiles).
        Only applies to meaningful numeric columns.
        """
        stats_summary = {}
        if self.df.empty:
            return stats_summary

        numeric_cols = [c for c in ["Vehicles_Involved", "Casualties", "Hour"] if c in self.df.columns]
        for col in numeric_cols:
            clean_vals = pd.to_numeric(self.df[col], errors="coerce").dropna().values
            if len(clean_vals) == 0:
                continue

            arr = clean_vals.astype(float)
            stats_summary[col] = {
                "Count": int(len(arr)),
                "Mean": round(float(np.mean(arr)), 3),
                "Median": round(float(np.median(arr)), 3),
                "Std Dev": round(float(np.std(arr, ddof=1)), 3) if len(arr) > 1 else 0.0,
                "Variance": round(float(np.var(arr, ddof=1)), 3) if len(arr) > 1 else 0.0,
                "Min": round(float(np.min(arr)), 3),
                "Max": round(float(np.max(arr)), 3),
                "25th %ile": round(float(np.percentile(arr, 25)), 3),
                "50th %ile": round(float(np.percentile(arr, 50)), 3),
                "75th %ile": round(float(np.percentile(arr, 75)), 3),
                "90th %ile": round(float(np.percentile(arr, 90)), 3),
            }

        return stats_summary

    def compute_correlation(self) -> Tuple[pd.DataFrame, bool]:
        """
        Calculates Pearson correlation matrix among numerical features.
        Returns (correlation_df, is_sufficient_numeric_columns).
        Includes statistical caveat: Correlation does not prove causation.
        """
        if self.df.empty:
            return pd.DataFrame(), False

        # Select valid numerical columns
        candidates = ["Vehicles_Involved", "Casualties", "Hour", "Latitude", "Longitude"]
        available_num_cols = []

        for col in candidates:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.dropna().nunique() > 1:
                    available_num_cols.append(col)

        if len(available_num_cols) < 2:
            return pd.DataFrame(), False

        sub_df = self.df[available_num_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub_df) < 5:
            return pd.DataFrame(), False

        corr_matrix = sub_df.corr(method="pearson").round(3)
        return corr_matrix, True

    # -------------------------------------------------------------
    # 9. Spatial / Geospatial Summary
    # -------------------------------------------------------------
    def get_spatial_summary(self) -> Dict[str, Any]:
        """
        Extracts spatial health metrics, bounding coordinates, centroid,
        severity counts on valid points, and top municipal clusters.
        """
        if self.df.empty or "Latitude" not in self.df.columns or "Longitude" not in self.df.columns:
            return {
                "has_coordinates": False,
                "valid_count": 0,
                "total_count": len(self.df),
                "valid_pct": 0.0,
                "lat_min": 0.0, "lat_max": 0.0,
                "lon_min": 0.0, "lon_max": 0.0,
                "centroid_lat": 0.0, "centroid_lon": 0.0,
                "severity_counts": {"Fatal": 0, "Serious": 0, "Minor": 0},
                "top_clusters": []
            }

        lat = pd.to_numeric(self.df["Latitude"], errors="coerce")
        lon = pd.to_numeric(self.df["Longitude"], errors="coerce")
        valid = lat.notna() & lon.notna() & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180) & ~((lat == 0) & (lon == 0))
        valid_df = self.df[valid].copy()

        tot = len(self.df)
        val_cnt = len(valid_df)
        pct = round(val_cnt / tot * 100, 1) if tot > 0 else 0.0

        if val_cnt == 0:
            return {
                "has_coordinates": False,
                "valid_count": 0,
                "total_count": tot,
                "valid_pct": 0.0,
                "lat_min": 0.0, "lat_max": 0.0,
                "lon_min": 0.0, "lon_max": 0.0,
                "centroid_lat": 0.0, "centroid_lon": 0.0,
                "severity_counts": {"Fatal": 0, "Serious": 0, "Minor": 0},
                "top_clusters": []
            }

        v_lat = pd.to_numeric(valid_df["Latitude"])
        v_lon = pd.to_numeric(valid_df["Longitude"])

        lat_min = round(float(v_lat.min()), 4)
        lat_max = round(float(v_lat.max()), 4)
        lon_min = round(float(v_lon.min()), 4)
        lon_max = round(float(v_lon.max()), 4)
        c_lat = round(float(v_lat.mean()), 4)
        c_lon = round(float(v_lon.mean()), 4)

        # Severity breakdown of valid GPS points
        sev_counts = {"Fatal": 0, "Serious": 0, "Minor": 0}
        if "Severity" in valid_df.columns:
            counts = valid_df["Severity"].astype(str).str.title().value_counts()
            for k in ["Fatal", "Serious", "Minor"]:
                sev_counts[k] = int(counts.get(k, 0))

        # Top clusters
        top_clusters = []
        if "City" in valid_df.columns:
            city_col = valid_df["City"].dropna().astype(str).str.strip()
            city_counts = city_col.value_counts().head(5)
            for city_name, cnt in city_counts.items():
                c_sub = valid_df[valid_df["City"].astype(str).str.strip() == city_name]
                mean_la = float(pd.to_numeric(c_sub["Latitude"]).mean()) if not c_sub.empty else 0.0
                mean_lo = float(pd.to_numeric(c_sub["Longitude"]).mean()) if not c_sub.empty else 0.0
                top_clusters.append({
                    "city": str(city_name),
                    "count": int(cnt),
                    "lat": round(mean_la, 3),
                    "lon": round(mean_lo, 3)
                })

        return {
            "has_coordinates": True,
            "valid_count": val_cnt,
            "total_count": tot,
            "valid_pct": pct,
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "centroid_lat": c_lat, "centroid_lon": c_lon,
            "severity_counts": sev_counts,
            "top_clusters": top_clusters
        }
