"""
Data Cleaning and Preprocessing Module for Traffic Accident Analyzer.
Provides missing value handling, duplicate removal, coordinate sanitization,
severity normalization, and auditable cleaning reports.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class DataCleaner:
    """Performs auditable dataset cleaning and quality assessment."""

    def __init__(self, original_df: Optional[pd.DataFrame] = None):
        self.original_df: Optional[pd.DataFrame] = original_df.copy() if original_df is not None else None
        self.cleaned_df: Optional[pd.DataFrame] = original_df.copy() if original_df is not None else None
        self.audit_log: List[str] = []

    def set_dataframe(self, df: pd.DataFrame):
        """Sets the working dataframe while retaining a snapshot of original data."""
        self.original_df = df.copy()
        self.cleaned_df = df.copy()
        self.audit_log.clear()

    def assess_quality(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Inspects data quality, returning comprehensive statistics on missing values,
        duplicates, invalid numeric columns, and coordinate validity.
        """
        target = self.cleaned_df if df is None else df
        if target is None or target.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "duplicates_count": 0,
                "missing_summary": {},
                "invalid_coords_count": 0,
                "invalid_numbers_count": 0
            }

        total_rows = len(target)
        duplicates_count = int(target.duplicated().sum())
        
        # Check if ID column has duplicate entries
        id_col = next((c for c in ["Accident_ID", "ID", "Crash_ID"] if c in target.columns), None)
        id_duplicates = int(target.duplicated(subset=[id_col]).sum()) if id_col else 0

        # Missing values summary
        missing_summary = {}
        for col in target.columns:
            cnt = int(target[col].isna().sum())
            pct = round((cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
            missing_summary[col] = {
                "count": cnt,
                "percentage": pct,
                "dtype": str(target[col].dtype),
                "valid_count": total_rows - cnt
            }

        # Invalid coordinates
        invalid_coords = 0
        if "Latitude" in target.columns and "Longitude" in target.columns:
            lat = pd.to_numeric(target["Latitude"], errors="coerce")
            lon = pd.to_numeric(target["Longitude"], errors="coerce")
            invalid_coords = int((
                lat.isna() | lon.isna() | 
                (lat < -90) | (lat > 90) | 
                (lon < -180) | (lon > 180) |
                ((lat == 0) & (lon == 0))
            ).sum())

        # Invalid numbers (negative vehicles or negative casualties)
        invalid_numbers = 0
        for col in ["Vehicles_Involved", "Casualties"]:
            if col in target.columns:
                nums = pd.to_numeric(target[col], errors="coerce")
                invalid_numbers += int((nums.isna() | (nums < 0)).sum())

        return {
            "total_rows": total_rows,
            "total_columns": len(target.columns),
            "duplicates_count": duplicates_count,
            "id_duplicates_count": id_duplicates,
            "missing_summary": missing_summary,
            "invalid_coords_count": invalid_coords,
            "invalid_numbers_count": invalid_numbers
        }

    def clean(
        self,
        drop_duplicates: bool = True,
        fill_missing_categorical: bool = True,
        categorical_fill_val: str = "Unknown",
        fill_missing_numeric: bool = True,
        numeric_fill_strategy: str = "median",  # "median" or "mean"
        sanitize_coordinates: bool = True,
        drop_invalid_coords: bool = False,
        normalize_severities: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes user-selected cleaning operations and logs every change made.
        Returns (cleaned_df, cleaning_summary_dict).
        """
        if self.original_df is None or self.original_df.empty:
            return pd.DataFrame(), {"status": "error", "message": "No dataset loaded to clean."}

        df = self.original_df.copy()
        initial_rows = len(df)
        log = []
        metrics = {
            "initial_rows": initial_rows,
            "duplicates_removed": 0,
            "missing_filled": 0,
            "invalid_coords_handled": 0,
            "invalid_numbers_fixed": 0,
            "final_rows": initial_rows
        }

        # 1. Remove Duplicates
        if drop_duplicates:
            before_dup = len(df)
            df = df.drop_duplicates()
            # If ID column exists, deduplicate by ID as well
            id_col = next((c for c in ["Accident_ID", "ID", "Crash_ID"] if c in df.columns), None)
            if id_col:
                df = df.drop_duplicates(subset=[id_col], keep="first")
            dups_removed = before_dup - len(df)
            metrics["duplicates_removed"] = dups_removed
            if dups_removed > 0:
                log.append(f"Removed {dups_removed} duplicate records.")
            else:
                log.append("No duplicate records found.")

        # 2. Normalize Severities
        if normalize_severities and "Severity" in df.columns:
            def norm_sev(val):
                if pd.isna(val):
                    return "Unknown"
                s = str(val).strip().lower()
                if "fatal" in s or s in ["1", 1]:
                    return "Fatal"
                if "serious" in s or "sever" in s or s in ["2", 2]:
                    return "Serious"
                if "minor" in s or "slight" in s or s in ["3", 3]:
                    return "Minor"
                return str(val).strip().title()

            df["Severity"] = df["Severity"].apply(norm_sev)
            log.append("Standardized Severity values to canonical categories (Fatal, Serious, Minor).")

        # 3. Handle Numerical Columns
        numeric_cols = [c for c in ["Vehicles_Involved", "Casualties", "Hour"] if c in df.columns]
        for col in numeric_cols:
            original_nas = df[col].isna().sum()
            # Coerce to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Fix negative values
            neg_mask = df[col] < 0
            neg_count = int(neg_mask.sum())
            if neg_count > 0:
                df.loc[neg_mask, col] = np.nan
                metrics["invalid_numbers_fixed"] += neg_count
                log.append(f"Replaced {neg_count} negative values in '{col}' with NaN.")

            if fill_missing_numeric:
                val = df[col].median() if numeric_fill_strategy == "median" else df[col].mean()
                if pd.isna(val):
                    val = 1.0  # sensible fallback
                fill_val = round(float(val), 1 if col != "Hour" else 0)
                filled_count = int(df[col].isna().sum())
                if filled_count > 0:
                    df[col] = df[col].fillna(fill_val)
                    metrics["missing_filled"] += filled_count
                    log.append(f"Imputed {filled_count} missing values in '{col}' with {numeric_fill_strategy} ({fill_val}).")

        # 4. Handle Categorical Columns
        categorical_cols = [c for c in ["City", "Weather", "Road_Surface", "Light_Condition"] if c in df.columns]
        if fill_missing_categorical:
            for col in categorical_cols:
                missing_cnt = int(df[col].isna().sum())
                if missing_cnt > 0:
                    df[col] = df[col].fillna(categorical_fill_val)
                    metrics["missing_filled"] += missing_cnt
                    log.append(f"Filled {missing_cnt} missing values in '{col}' with '{categorical_fill_val}'.")

        # 5. Sanitize Coordinates
        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
            
            invalid_mask = (
                df["Latitude"].isna() | df["Longitude"].isna() |
                (df["Latitude"] < -90) | (df["Latitude"] > 90) |
                (df["Longitude"] < -180) | (df["Longitude"] > 180) |
                ((df["Latitude"] == 0) & (df["Longitude"] == 0))
            )
            invalid_cnt = int(invalid_mask.sum())
            if invalid_cnt > 0:
                metrics["invalid_coords_handled"] = invalid_cnt
                if drop_invalid_coords:
                    df = df[~invalid_mask]
                    log.append(f"Dropped {invalid_cnt} records with out-of-bounds or zero coordinates.")
                elif sanitize_coordinates:
                    # Set invalid coordinates to NaN so geospatial viewer can safely skip them
                    df.loc[invalid_mask, ["Latitude", "Longitude"]] = np.nan
                    log.append(f"Set {invalid_cnt} invalid coordinate pairs to NaN (preserving non-spatial record data).")

        # 6. Recompute Date/Hour fields if Date/Time present
        if "Date" in df.columns:
            try:
                parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
                df["Parsed_Date"] = parsed_dates
                df["Month"] = parsed_dates.dt.strftime("%b")
                df["Year"] = parsed_dates.dt.year
                df["Weekday"] = parsed_dates.dt.day_name()
                log.append("Re-synchronized temporal features (Month, Weekday, Year).")
            except Exception:
                pass

        metrics["final_rows"] = len(df)
        self.cleaned_df = df
        self.audit_log = log

        return df, {
            "status": "success",
            "metrics": metrics,
            "audit_log": log
        }

    def restore_original(self) -> pd.DataFrame:
        """Restores the original unmodified dataset."""
        if self.original_df is not None:
            self.cleaned_df = self.original_df.copy()
            self.audit_log = ["Dataset restored to original imported state."]
            return self.cleaned_df
        return pd.DataFrame()
