"""
Reports and Export Module for Traffic Accident Analyzer.
Generates comprehensive academic analytics reports, statistical summaries,
and exports data tables to CSV and formatted text files.
"""
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .analysis import AccidentAnalyzer


class ReportGenerator:
    """Creates formatted text and markdown reports, and handles table exports."""

    @staticmethod
    def generate_text_report(
        analyzer: AccidentAnalyzer,
        dataset_name: str = "Dataset",
        active_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Builds a comprehensive, professionally formatted academic report of the current dataset.
        Includes KPIs, severity analysis, time patterns, location ranking, weather correlations,
        numerical statistics, and formal academic disclaimers.
        """
        df = analyzer.df
        kpis = analyzer.get_kpis()
        sev_data = analyzer.analyze_severity()
        time_data = analyzer.analyze_time()
        loc_data = analyzer.analyze_locations(top_n=10)
        weather_data = analyzer.analyze_weather()
        road_data = analyzer.analyze_road_and_light()
        stats_data = analyzer.compute_statistics()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine date range covered
        date_range_str = "Not available"
        if "Parsed_Date" in df.columns and not df["Parsed_Date"].dropna().empty:
            min_date = df["Parsed_Date"].dropna().min().strftime("%Y-%m-%d")
            max_date = df["Parsed_Date"].dropna().max().strftime("%Y-%m-%d")
            date_range_str = f"{min_date} to {max_date}"
        elif "Date" in df.columns and not df["Date"].dropna().empty:
            date_range_str = f"{df['Date'].dropna().iloc[0]} to {df['Date'].dropna().iloc[-1]}"

        # Format filter summary
        filter_lines = []
        if active_filters:
            for k, v in active_filters.items():
                if v and v != "All":
                    filter_lines.append(f"  - {k}: {v}")
        filter_str = "\n".join(filter_lines) if filter_lines else "  None (Full Dataset)"

        lines = [
            "=" * 78,
            "               TRAFFIC ACCIDENT ANALYZER - ANALYTICAL REPORT",
            "                Road Safety Data Analytics & Pattern Insights",
            "=" * 78,
            f"Generated On  : {now_str}",
            f"Dataset Name  : {os.path.basename(dataset_name)}",
            f"Date Coverage : {date_range_str}",
            f"Total Records : {kpis['total_accidents']:,}",
            f"Active Filters:\n{filter_str}",
            "-" * 78,
            "",
            "1. EXECUTIVE SUMMARY & KEY PERFORMANCE INDICATORS",
            "-" * 50,
            f"  * Total Recorded Incidents : {kpis['total_accidents']:,}",
            f"  * Fatal Accidents          : {kpis['fatal_accidents']:,} ({kpis['fatal_pct']}%)",
            f"  * Serious Accidents        : {kpis['serious_accidents']:,} ({kpis['serious_pct']}%)",
            f"  * Minor Accidents          : {kpis['minor_accidents']:,} ({kpis['minor_pct']}%)",
            f"  * Peak Accident Hour       : {kpis['peak_hour']} (Recorded: {kpis['peak_hour_count']} incidents)",
            f"  * Unique Cities / Locations: {kpis['unique_locations']}",
            f"  * Average Vehicles Involved: {kpis['avg_vehicles']}",
            "",
            "2. ACCIDENT SEVERITY DISTRIBUTION",
            "-" * 50,
            f"  {'Severity Category':<20} {'Count':<10} {'Percentage':<12}",
            f"  {'-'*18:<20} {'-'*8:<10} {'-'*10:<12}"
        ]

        if not sev_data["counts"].empty:
            for cat, cnt in sev_data["counts"].items():
                pct = sev_data["percentages"].get(cat, 0.0)
                lines.append(f"  {cat:<20} {cnt:<10} {pct:>8.2f}%")
        else:
            lines.append("  No severity categories recorded.")

        lines.extend([
            "",
            "3. TEMPORAL & TIME-SERIES PATTERNS",
            "-" * 50,
            f"  * Peak Hour of the Day: {kpis['peak_hour']}",
            f"  * Peak Day of the Week: {time_data.get('peak_weekday') or 'N/A'}",
            f"  * Peak Month of Year  : {time_data.get('peak_month') or 'N/A'} ({time_data.get('peak_month_count', 0)} incidents)",
            "",
            "  Hourly Breakdown (00:00 to 23:00):"
        ])

        hourly = time_data["hourly_counts"]
        # Print hourly in a compact 4-column format
        hourly_chunks = []
        for h in range(24):
            hourly_chunks.append(f"{h:02d}:00={hourly.get(h, 0):<3}")
        for i in range(0, 24, 6):
            lines.append("    " + "  |  ".join(hourly_chunks[i:i+6]))

        lines.extend([
            "",
            "4. TOP 10 ACCIDENT LOCATIONS / HOTSPOTS",
            "-" * 50,
            f"  {'Rank':<6} {'Location / City':<25} {'Accidents':<12} {'Share (%)':<10}",
            f"  {'-'*4:<6} {'-'*23:<25} {'-'*10:<12} {'-'*8:<10}"
        ])

        top_locs = loc_data["top_locations"]
        if not top_locs.empty:
            tot = kpis["total_accidents"]
            for idx, (city, cnt) in enumerate(top_locs.items(), start=1):
                pct = (cnt / tot * 100) if tot > 0 else 0
                lines.append(f"  #{idx:<5} {city:<25} {cnt:<12} {pct:>7.2f}%")
        else:
            lines.append("  No location data available.")

        lines.extend([
            "",
            "5. WEATHER & ENVIRONMENTAL CONDITIONS",
            "-" * 50,
            f"  {'Condition':<22} {'Accidents':<12} {'Share (%)':<10}",
            f"  {'-'*20:<22} {'-'*10:<12} {'-'*8:<10}"
        ])

        weather_counts = weather_data["weather_counts"]
        if not weather_counts.empty:
            tot = kpis["total_accidents"]
            for cond, cnt in weather_counts.items():
                pct = (cnt / tot * 100) if tot > 0 else 0
                lines.append(f"  {cond:<22} {cnt:<12} {pct:>7.2f}%")
        else:
            lines.append("  No weather condition records available.")

        lines.extend([
            "",
            "6. ROAD SURFACE & LIGHTING CONDITIONS",
            "-" * 50
        ])
        road_counts = road_data["road_surface_counts"]
        if not road_counts.empty:
            lines.append("  Road Surface Breakdown:")
            for srf, cnt in road_counts.items():
                lines.append(f"    - {srf:<20}: {cnt} incidents")
        light_counts = road_data["light_condition_counts"]
        if not light_counts.empty:
            lines.append("  Lighting Conditions:")
            for lgt, cnt in light_counts.items():
                lines.append(f"    - {lgt:<25}: {cnt} incidents")

        lines.extend([
            "",
            "7. NUMERICAL & STATISTICAL ANALYSIS (NUMPY)",
            "-" * 50
        ])

        if stats_data:
            for col_name, s in stats_data.items():
                lines.extend([
                    f"  Column: '{col_name}' (N={s['Count']})",
                    f"    Mean: {s['Mean']:<8} | Median: {s['Median']:<8} | Std Dev: {s['Std Dev']:<8} | Variance: {s['Variance']}",
                    f"    Min : {s['Min']:<8} | Max   : {s['Max']:<8} | 25th %: {s['25th %ile']:<8} | 75th %: {s['75th %ile']}",
                    ""
                ])
        else:
            lines.append("  No numeric columns available for statistical summary.")

        # Academic complexity and disclaimer
        lines.extend([
            "8. DESIGN & ANALYSIS OF ALGORITHMS (ACADEMIC COMPLEXITY)",
            "-" * 50,
            "  * Linear Dataset Filtering      : O(N) Time, O(N) Space",
            "  * Hash-based Grouping (GroupBy) : O(N) Average Time, O(U) Space (U <= N unique keys)",
            "  * Top-K Location Sorting        : O(U log U) Time using Timsort",
            "  * Vectorized Statistical Pass   : O(N) Time, O(1) Auxiliary Space",
            "  * Pearson Correlation Matrix    : O(M^2 * N) Time (M numeric features, N records)",
            "",
            "=" * 78,
            "                            ACADEMIC DISCLAIMER",
            "=" * 78,
            "  This software application and the resulting report are designed for educational",
            "  and academic data analytics purposes. All statistical findings represent historical",
            "  or benchmark descriptive patterns and do not predict individual accident causes",
            "  or guarantee road safety outcomes. Correlation does not imply causation.",
            "=" * 78
        ])

        return "\n".join(lines)

    @staticmethod
    def export_text_report(report_text: str, file_path: str) -> Tuple[bool, Optional[str]]:
        """Saves generated report text to disk."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            return True, None
        except Exception as e:
            return False, f"Failed to save report: {str(e)}"

    @staticmethod
    def export_summary_tables(analyzer: AccidentAnalyzer, output_dir: str) -> Tuple[bool, List[str], Optional[str]]:
        """
        Exports summary CSV tables (Severity, Locations, Hourly) to an output directory.
        Returns (success_bool, list_of_saved_files, error_message).
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_files = []

            # 1. Severity Summary
            sev_data = analyzer.analyze_severity()
            if not sev_data["counts"].empty:
                sev_df = pd.DataFrame({
                    "Severity": sev_data["counts"].index,
                    "Count": sev_data["counts"].values,
                    "Percentage": [sev_data["percentages"].get(c, 0.0) for c in sev_data["counts"].index]
                })
                sev_path = os.path.join(output_dir, "severity_summary.csv")
                sev_df.to_csv(sev_path, index=False)
                saved_files.append(sev_path)

            # 2. Locations Summary
            loc_data = analyzer.analyze_locations(top_n=50)
            if not loc_data["all_locations"].empty:
                loc_df = pd.DataFrame({
                    "Location": loc_data["all_locations"].index,
                    "Accident_Count": loc_data["all_locations"].values
                })
                loc_path = os.path.join(output_dir, "locations_summary.csv")
                loc_df.to_csv(loc_path, index=False)
                saved_files.append(loc_path)

            # 3. Hourly Summary
            time_data = analyzer.analyze_time()
            hourly_s = time_data["hourly_counts"]
            hour_df = pd.DataFrame({
                "Hour": [f"{h:02d}:00" for h in range(24)],
                "Accident_Count": [hourly_s.get(h, 0) for h in range(24)]
            })
            hour_path = os.path.join(output_dir, "hourly_summary.csv")
            hour_df.to_csv(hour_path, index=False)
            saved_files.append(hour_path)

            return True, saved_files, None
        except Exception as e:
            return False, [], f"Export tables failed: {str(e)}"
