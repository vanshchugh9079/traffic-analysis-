"""
Data Loading and Normalization Module for Traffic Accident Analyzer.
Handles CSV and Excel loading, flexible column mapping, date parsing, and filtering.
"""
import os
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class DataLoader:
    """Manages dataset importing, normalization, filtering, and export."""

    # Canonical column name definitions with common aliases (in lowercase)
    COLUMN_ALIASES: Dict[str, List[str]] = {
        "Accident_ID": [
            "accident_id", "accident_index", "id", "crash_id", "incident_id",
            "case_id", "collision_id", "record_id", "report_num"
        ],
        "Date": [
            "date", "accident_date", "crash_date", "datetime", "timestamp",
            "incident_date", "date_of_accident", "collision_date"
        ],
        "Time": [
            "time", "accident_time", "crash_time", "time_of_day",
            "incident_time", "time_of_accident", "collision_time"
        ],
        "City": [
            "city", "location", "district", "area", "town", "county",
            "place", "borough", "municipality", "suburb", "accident_location"
        ],
        "Severity": [
            "severity", "accident_severity", "crash_severity", "severity_level",
            "injury_severity", "collision_severity", "accident_level"
        ],
        "Weather": [
            "weather", "weather_condition", "weather_conditions", "climate",
            "atmospheric_conditions", "weather_state"
        ],
        "Road_Surface": [
            "road_surface", "road_surface_condition", "road_surface_conditions",
            "surface_condition", "road_condition", "surface_type"
        ],
        "Light_Condition": [
            "light_condition", "lighting", "light_conditions", "illumination",
            "lighting_condition", "light_state"
        ],
        "Vehicles_Involved": [
            "vehicles_involved", "number_of_vehicles", "vehicles", "vehicle_count",
            "no_of_vehicles", "tot_vehicles", "num_vehicles"
        ],
        "Casualties": [
            "casualties", "number_of_casualties", "injuries", "casualty_count",
            "total_injuries", "num_casualties", "injured_count"
        ],
        "Latitude": ["latitude", "lat", "y_coord", "coord_y", "y"],
        "Longitude": ["longitude", "long", "lon", "lng", "x_coord", "coord_x", "x"]
    }

    SEVERITY_MAPPING = {
        "1": "Fatal",
        1: "Fatal",
        "fatal": "Fatal",
        "fatal injury": "Fatal",
        "killed": "Fatal",
        "deadly": "Fatal",
        "2": "Serious",
        2: "Serious",
        "serious": "Serious",
        "severe": "Serious",
        "major": "Serious",
        "hospitalised": "Serious",
        "incapacitating": "Serious",
        "3": "Minor",
        3: "Minor",
        "minor": "Minor",
        "slight": "Minor",
        "moderate": "Minor",
        "non-incapacitating": "Minor"
    }

    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.current_data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.column_mapping: Dict[str, str] = {}

    def load_file(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load CSV or Excel file safely with automatic encoding and format detection.
        Returns (DataFrame, error_message).
        """
        if not os.path.exists(file_path):
            return None, f"File does not exist: {file_path}"

        ext = os.path.splitext(file_path)[1].lower()
        df = None
        error_msg = None

        try:
            if ext == ".csv":
                # Try common encodings
                for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                if df is None:
                    return None, "Failed to decode CSV file with supported encodings (utf-8, latin1, cp1252)."
            elif ext in [".xlsx", ".xls"]:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl" if ext == ".xlsx" else None)
                except Exception as excel_err:
                    return None, f"Excel import error: {str(excel_err)}"
            else:
                return None, f"Unsupported file extension '{ext}'. Please select a CSV or Excel (.xlsx, .xls) file."

            if df is None or df.empty:
                return None, "The selected file is empty."

            # Normalize column names and format types
            df, mapping = self.normalize_columns(df)
            self.column_mapping = mapping
            
            # Post-process standard columns (Date, Time, Severity, Numbers)
            df = self._post_process_dataset(df)

            self.raw_data = df.copy()
            self.current_data = df.copy()
            self.file_path = file_path
            return self.current_data, None

        except Exception as ex:
            return None, f"Error loading file: {str(ex)}"

    def normalize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Maps raw column names to standard canonical names based on known aliases.
        Preserves unmapped columns as-is.
        """
        mapping = {}
        cleaned_cols = {}
        for col in df.columns:
            # Clean string for comparison
            normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip().lower())
            mapped_name = col
            for canonical, aliases in self.COLUMN_ALIASES.items():
                if normalized in aliases or str(col).strip().lower() == canonical.lower():
                    mapped_name = canonical
                    break
            cleaned_cols[col] = mapped_name
            mapping[str(col)] = mapped_name

        renamed_df = df.rename(columns=cleaned_cols)
        return renamed_df, mapping

    def _post_process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts dates, standardizes severities, extracts hours, and ensures
        sensible data types.
        """
        processed = df.copy()

        # Normalize Severity if present
        if "Severity" in processed.columns:
            processed["Severity"] = processed["Severity"].apply(self._normalize_severity_value)

        # Parse Date and Time
        if "Date" in processed.columns:
            try:
                parsed_dates = pd.to_datetime(processed["Date"], errors="coerce")
                processed["Parsed_Date"] = parsed_dates
                if "Month" not in processed.columns:
                    processed["Month"] = parsed_dates.dt.strftime("%b")
                if "Year" not in processed.columns:
                    processed["Year"] = parsed_dates.dt.year
                if "Weekday" not in processed.columns:
                    processed["Weekday"] = parsed_dates.dt.day_name()
            except Exception:
                pass

        # Parse Time and Hour
        if "Time" in processed.columns:
            try:
                # Handle HH:MM or HH:MM:SS or integer hour
                def extract_hour(val):
                    if pd.isna(val):
                        return np.nan
                    val_str = str(val).strip()
                    # Check if 'HH:MM'
                    if ":" in val_str:
                        parts = val_str.split(":")
                        return int(parts[0]) % 24
                    elif val_str.isdigit():
                        h = int(val_str)
                        return h if 0 <= h < 24 else np.nan
                    return np.nan

                if "Hour" not in processed.columns:
                    processed["Hour"] = processed["Time"].apply(extract_hour)
            except Exception:
                pass
        elif "Parsed_Date" in processed.columns and "Hour" not in processed.columns:
            # Maybe datetime contained hour
            processed["Hour"] = processed["Parsed_Date"].dt.hour

        # Ensure numeric columns are properly typed
        for num_col in ["Vehicles_Involved", "Casualties", "Hour"]:
            if num_col in processed.columns:
                processed[num_col] = pd.to_numeric(processed[num_col], errors="coerce")

        for coord_col in ["Latitude", "Longitude"]:
            if coord_col in processed.columns:
                processed[coord_col] = pd.to_numeric(processed[coord_col], errors="coerce")

        return processed

    def _normalize_severity_value(self, val: Any) -> str:
        """Helper to map any severity value to Fatal, Serious, or Minor."""
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip().lower()
        if s in self.SEVERITY_MAPPING:
            return self.SEVERITY_MAPPING[s]
        # Check partial match
        if "fatal" in s or "death" in s or "kill" in s:
            return "Fatal"
        if "serio" in s or "sever" in s or "major" in s or "hospital" in s:
            return "Serious"
        if "mino" in s or "slight" in s or "mod" in s:
            return "Minor"
        return str(val).strip().title()

    def get_metadata(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Returns structural metadata and statistics about the dataset."""
        target_df = self.current_data if df is None else df
        if target_df is None or target_df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "columns": [],
                "dtypes": {},
                "missing_counts": {},
                "file_path": self.file_path or "None"
            }

        return {
            "total_rows": len(target_df),
            "total_columns": len(target_df.columns),
            "columns": list(target_df.columns),
            "dtypes": {col: str(target_df[col].dtype) for col in target_df.columns},
            "missing_counts": target_df.isna().sum().to_dict(),
            "file_path": self.file_path or "In-memory dataset"
        }

    def filter_data(
        self,
        df: Optional[pd.DataFrame] = None,
        city: Optional[str] = None,
        severity: Optional[str] = None,
        weather: Optional[str] = None,
        road_surface: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Applies multi-criteria filters to the dataset.
        Returns a filtered view without altering the original imported data.
        """
        base_df = self.raw_data if df is None else df
        if base_df is None or base_df.empty:
            return pd.DataFrame()

        filtered = base_df.copy()

        # Filter by City
        if city and city != "All" and "City" in filtered.columns:
            filtered = filtered[filtered["City"].astype(str).str.strip().str.lower() == str(city).strip().lower()]

        # Filter by Severity
        if severity and severity != "All" and "Severity" in filtered.columns:
            filtered = filtered[filtered["Severity"].astype(str).str.strip().str.lower() == str(severity).strip().lower()]

        # Filter by Weather
        if weather and weather != "All" and "Weather" in filtered.columns:
            filtered = filtered[filtered["Weather"].astype(str).str.strip().str.lower() == str(weather).strip().lower()]

        # Filter by Road Surface
        if road_surface and road_surface != "All" and "Road_Surface" in filtered.columns:
            filtered = filtered[filtered["Road_Surface"].astype(str).str.strip().str.lower() == str(road_surface).strip().lower()]

        # Filter by Date range
        if "Parsed_Date" in filtered.columns:
            if start_date:
                try:
                    s_date = pd.to_datetime(start_date)
                    filtered = filtered[filtered["Parsed_Date"] >= s_date]
                except Exception:
                    pass
            if end_date:
                try:
                    e_date = pd.to_datetime(end_date)
                    filtered = filtered[filtered["Parsed_Date"] <= e_date]
                except Exception:
                    pass

        # Text search across string columns
        if search_query and str(search_query).strip():
            query = str(search_query).strip().lower()
            mask = pd.Series(False, index=filtered.index)
            # Check string / object columns
            for col in filtered.columns:
                if filtered[col].dtype == object or str(filtered[col].dtype).startswith("string"):
                    col_mask = filtered[col].astype(str).str.lower().str.contains(query, regex=False, na=False)
                    mask = mask | col_mask
                elif "Accident_ID" == col or "City" == col:
                    col_mask = filtered[col].astype(str).str.lower().str.contains(query, regex=False, na=False)
                    mask = mask | col_mask
            filtered = filtered[mask]

        self.current_data = filtered
        return filtered

    def export_to_csv(self, df: pd.DataFrame, file_path: str) -> Tuple[bool, Optional[str]]:
        """Exports dataframe to CSV. Returns (success_bool, error_msg)."""
        try:
            # Exclude internal helper columns like Parsed_Date if desired, or export clean view
            export_df = df.copy()
            if "Parsed_Date" in export_df.columns:
                export_df = export_df.drop(columns=["Parsed_Date"])
            export_df.to_csv(file_path, index=False)
            return True, None
        except Exception as e:
            return False, f"Export failed: {str(e)}"
