"""
Visualizations Module for Traffic Accident Analyzer.
Embeds high-resolution, interactive Matplotlib charts into Tkinter interfaces.
Uses an object-oriented approach with a modern Navy, Blue, and White palette.
"""
from typing import Optional, Tuple
import matplotlib
# Use Agg or TkAgg depending on headless vs GUI
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from .analysis import AccidentAnalyzer


class Visualizer:
    """Generates styled Matplotlib figures for desktop embedding and image export."""

    # Color Palette matching modern Navy/Blue/White theme
    THEME = {
        "bg": "#FFFFFF",
        "card_bg": "#F8FAFC",
        "text": "#0F172A",
        "muted_text": "#64748B",
        "grid": "#E2E8F0",
        "primary_blue": "#2563EB",
        "sky_blue": "#0284C7",
        "navy": "#1E293B",
        "fatal_red": "#EF4444",
        "serious_amber": "#F59E0B",
        "minor_green": "#10B981",
        "palette_bars": ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#CBD5E1"]
    }

    SEVERITY_COLORS = {
        "Fatal": "#EF4444",
        "Serious": "#F59E0B",
        "Minor": "#10B981",
        "Unknown": "#94A3B8"
    }

    @staticmethod
    def _apply_axes_styling(ax, title: str, xlabel: str = "", ylabel: str = ""):
        """Applies consistent typography, grid styling, and border cleanup."""
        ax.set_facecolor(Visualizer.THEME["bg"])
        ax.set_title(title, fontsize=12, fontweight="bold", color=Visualizer.THEME["text"], pad=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold", color=Visualizer.THEME["muted_text"], labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color=Visualizer.THEME["muted_text"], labelpad=8)
        ax.tick_params(colors=Visualizer.THEME["muted_text"], labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.5, color=Visualizer.THEME["grid"])
        # Remove top and right spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(Visualizer.THEME["grid"])
            ax.spines[spine].set_linewidth(1.0)

    @staticmethod
    def _draw_empty_message(ax, message: str = "No data available for the active filters."):
        """Renders a polite empty-state banner when data is unavailable."""
        ax.clear()
        ax.set_facecolor(Visualizer.THEME["bg"])
        ax.axis("off")
        ax.text(
            0.5, 0.5, f"ℹ️ {message}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=Visualizer.THEME["muted_text"],
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.8)
        )

    # -------------------------------------------------------------
    # 1. Severity Bar Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_severity_bar(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots accident severity distribution with count and percentage annotations."""
        if fig is None:
            fig = Figure(figsize=(5.5, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        data = analyzer.analyze_severity()
        counts = data["counts"]

        if counts.empty:
            cls._draw_empty_message(ax, "No severity records to display.")
            fig.tight_layout()
            return fig

        categories = list(counts.index)
        values = list(counts.values)
        total = data["total"]
        colors = [cls.SEVERITY_COLORS.get(cat, cls.THEME["primary_blue"]) for cat in categories]

        bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none", zorder=3)
        cls._apply_axes_styling(ax, "Accident Severity Breakdown", xlabel="Severity Level", ylabel="Recorded Accidents")

        # Add data labels on top of bars
        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.18)
        for bar in bars:
            h = bar.get_height()
            pct = (h / total * 100) if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + (max_val * 0.02),
                f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=cls.THEME["text"]
            )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 2. Severity Donut / Pie Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_severity_pie(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots severity proportion as an elegant donut chart."""
        if fig is None:
            fig = Figure(figsize=(5.5, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        data = analyzer.analyze_severity()
        counts = data["counts"]

        if counts.empty:
            cls._draw_empty_message(ax, "No severity records to display.")
            fig.tight_layout()
            return fig

        labels = list(counts.index)
        values = list(counts.values)
        colors = [cls.SEVERITY_COLORS.get(cat, cls.THEME["primary_blue"]) for cat in labels]

        # Draw donut chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
        )

        for text in texts:
            text.set_color(cls.THEME["text"])
            text.set_fontsize(9)
            text.set_fontweight("semibold")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(8.5)
            autotext.set_fontweight("bold")

        ax.set_title("Accident Severity Proportion", fontsize=12, fontweight="bold", color=cls.THEME["text"], pad=14)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 3. Hourly Trend Line Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_hourly_trend(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots 24-hour accident distribution with area highlight and peak marker."""
        if fig is None:
            fig = Figure(figsize=(6.5, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        time_data = analyzer.analyze_time()
        hourly = time_data["hourly_counts"]

        if hourly.sum() == 0:
            cls._draw_empty_message(ax, "No hourly time data available.")
            fig.tight_layout()
            return fig

        hours = list(range(24))
        counts = [hourly.get(h, 0) for h in hours]

        # Line plot with gradient fill
        ax.plot(hours, counts, color=cls.THEME["primary_blue"], linewidth=2.5, marker="o", markersize=4, zorder=4)
        ax.fill_between(hours, counts, color=cls.THEME["primary_blue"], alpha=0.15, zorder=2)

        # Highlight peak hour
        peak_h = time_data["peak_hour"]
        if peak_h is not None and hourly[peak_h] > 0:
            peak_val = hourly[peak_h]
            ax.scatter([peak_h], [peak_val], color=cls.THEME["fatal_red"], s=60, zorder=5)
            ax.annotate(
                f"Peak: {peak_h:02d}:00 ({peak_val})",
                xy=(peak_h, peak_val),
                xytext=(peak_h + 0.5, peak_val + max(counts) * 0.08),
                fontsize=8.5, fontweight="bold", color=cls.THEME["fatal_red"],
                arrowprops=dict(arrowstyle="->", color=cls.THEME["fatal_red"], lw=1.2)
            )

        cls._apply_axes_styling(ax, "Accidents by Hour of the Day (24-Hour Cycle)", xlabel="Hour of Day", ylabel="Accidents")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=30)
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 5)

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 4. Monthly Trend Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_monthly_trend(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots monthly accident progression."""
        if fig is None:
            fig = Figure(figsize=(6.0, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        time_data = analyzer.analyze_time()
        monthly = time_data["monthly_counts"]

        if monthly.empty or monthly.sum() == 0:
            cls._draw_empty_message(ax, "No monthly date information available.")
            fig.tight_layout()
            return fig

        months = list(monthly.index)
        values = list(monthly.values)

        bars = ax.bar(months, values, color=cls.THEME["sky_blue"], width=0.6, zorder=3)
        cls._apply_axes_styling(ax, "Monthly Accident Frequency", xlabel="Month", ylabel="Accidents")

        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.2)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + (max_val * 0.02),
                    f"{int(h)}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=cls.THEME["text"]
                )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 5. Top 10 Locations Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_top_locations(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None, top_n: int = 10) -> Figure:
        """Plots horizontal bar chart ranking top accident locations."""
        if fig is None:
            fig = Figure(figsize=(6.2, 4.0), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        loc_data = analyzer.analyze_locations(top_n=top_n)
        top_locs = loc_data["top_locations"]

        if top_locs.empty:
            cls._draw_empty_message(ax, "No location records available.")
            fig.tight_layout()
            return fig

        # Reverse to show highest on top
        cities = list(reversed(top_locs.index))
        counts = list(reversed(top_locs.values))

        y_pos = np.arange(len(cities))
        bars = ax.barh(y_pos, counts, color=cls.THEME["navy"], height=0.6, zorder=3)

        cls._apply_axes_styling(ax, f"Top {len(cities)} Accident Locations", xlabel="Recorded Accidents", ylabel="")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cities, fontsize=9, fontweight="semibold", color=cls.THEME["text"])

        max_val = max(counts) if counts else 1
        ax.set_xlim(0, max_val * 1.18)
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + (max_val * 0.02), bar.get_y() + bar.get_height() / 2,
                f"{int(w)}", ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=cls.THEME["text"]
            )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 6. Weather Conditions vs Severity Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_weather_distribution(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots accident counts grouped by weather condition."""
        if fig is None:
            fig = Figure(figsize=(6.0, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        weather_data = analyzer.analyze_weather()
        counts = weather_data["weather_counts"]

        if counts.empty:
            cls._draw_empty_message(ax, "No weather condition records available.")
            fig.tight_layout()
            return fig

        conditions = list(counts.index)
        values = list(counts.values)

        palette = ["#0284C7", "#2563EB", "#6366F1", "#8B5CF6", "#EC4899", "#64748B"]
        colors = [palette[i % len(palette)] for i in range(len(conditions))]

        bars = ax.bar(conditions, values, color=colors, width=0.55, zorder=3)
        cls._apply_axes_styling(ax, "Accidents by Weather Condition", xlabel="Weather Condition", ylabel="Accidents")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=20, ha="right")

        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.2)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + (max_val * 0.02),
                f"{int(h)}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=cls.THEME["text"]
            )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 7. Road Surface Conditions Chart
    # -------------------------------------------------------------
    @classmethod
    def plot_road_surface(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots accident counts categorized by road surface condition."""
        if fig is None:
            fig = Figure(figsize=(6.0, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        road_data = analyzer.analyze_road_and_light()
        counts = road_data["road_surface_counts"]

        if counts.empty:
            cls._draw_empty_message(ax, "No road surface records available.")
            fig.tight_layout()
            return fig

        surfaces = list(counts.index)
        values = list(counts.values)

        bars = ax.bar(surfaces, values, color="#3B82F6", width=0.55, zorder=3)
        cls._apply_axes_styling(ax, "Accidents by Road Surface Condition", xlabel="Road Surface", ylabel="Accidents")
        ax.set_xticks(range(len(surfaces)))
        ax.set_xticklabels(surfaces, rotation=15, ha="right")

        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.2)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + (max_val * 0.02),
                f"{int(h)}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=cls.THEME["text"]
            )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 8. Vehicle Distribution Histogram
    # -------------------------------------------------------------
    @classmethod
    def plot_vehicle_distribution(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots frequency distribution of vehicles involved per accident."""
        if fig is None:
            fig = Figure(figsize=(5.8, 3.8), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        veh_data = analyzer.analyze_vehicles()
        dist = veh_data["distribution"]

        if dist.empty:
            cls._draw_empty_message(ax, "No vehicle count data available.")
            fig.tight_layout()
            return fig

        veh_counts = [str(x) for x in dist.index]
        freqs = list(dist.values)

        bars = ax.bar(veh_counts, freqs, color="#6366F1", width=0.5, zorder=3)
        cls._apply_axes_styling(ax, "Distribution of Vehicles Involved", xlabel="Number of Vehicles", ylabel="Accident Count")

        max_val = max(freqs) if freqs else 1
        ax.set_ylim(0, max_val * 1.22)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + (max_val * 0.02),
                f"{int(h)}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=cls.THEME["text"]
            )

        # Add summary stats box in corner
        stats_text = f"Mean: {veh_data['mean']}\nMedian: {veh_data['median']}\nStd Dev: {veh_data['std']}"
        ax.text(
            0.95, 0.90, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9)
        )

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 9. Correlation Heatmap
    # -------------------------------------------------------------
    @classmethod
    def plot_correlation_heatmap(cls, analyzer: AccidentAnalyzer, fig: Optional[Figure] = None) -> Figure:
        """Plots Pearson correlation heatmap with numerical values and academic disclaimer."""
        if fig is None:
            fig = Figure(figsize=(6.2, 4.2), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        corr_matrix, is_sufficient = analyzer.compute_correlation()

        if not is_sufficient or corr_matrix.empty:
            cls._draw_empty_message(ax, "Insufficient numeric columns to compute correlation matrix.\n(Requires at least 2 distinct numeric columns).")
            fig.tight_layout()
            return fig

        cols = list(corr_matrix.columns)
        matrix_vals = corr_matrix.values

        cax = ax.matshow(matrix_vals, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="left", fontsize=8.5, color=cls.THEME["text"])
        ax.set_yticklabels(cols, fontsize=8.5, color=cls.THEME["text"])

        # Annotate correlation numbers
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = matrix_vals[i, j]
                color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

        ax.set_title("Pearson Correlation Heatmap\n(Note: Correlation does not imply causation)",
                     fontsize=11, fontweight="bold", color=cls.THEME["text"], pad=16)

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------
    # 10. Geographic Coordinates Hotspot Scatter Plot
    # -------------------------------------------------------------
    @classmethod
    def plot_geospatial_hotspots(
        cls,
        analyzer: AccidentAnalyzer,
        fig: Optional[Figure] = None,
        city_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Figure:
        """
        Plots scatter coordinates of accidents color-coded by severity.
        Supports municipality zooming, cluster centroid annotations, and
        guarantees proper bounds fitting without overflow.
        """
        if fig is None:
            fig = Figure(figsize=(6.5, 4.2), dpi=100)
            fig.patch.set_facecolor(cls.THEME["bg"])
        else:
            fig.clear()

        ax = fig.add_subplot(111)
        df = analyzer.df

        if df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
            cls._draw_empty_message(ax, "No latitude and longitude coordinates available in dataset.")
            fig.tight_layout(pad=1.2)
            return fig

        lat = pd.to_numeric(df["Latitude"], errors="coerce")
        lon = pd.to_numeric(df["Longitude"], errors="coerce")
        valid = (
            lat.notna() & lon.notna() &
            (lat >= -90) & (lat <= 90) &
            (lon >= -180) & (lon <= 180) &
            ~((lat == 0) & (lon == 0))
        )
        valid_df = df[valid].copy()

        # Apply specific city filter if requested
        if city_filter and str(city_filter).strip() and str(city_filter).lower() != "all" and "City" in valid_df.columns:
            valid_df = valid_df[valid_df["City"].astype(str).str.strip().str.lower() == str(city_filter).strip().lower()]

        # Apply specific severity filter if requested
        if severity_filter and str(severity_filter).strip() and str(severity_filter).lower() != "all" and "Severity" in valid_df.columns:
            valid_df = valid_df[valid_df["Severity"].astype(str).str.strip().str.lower() == str(severity_filter).strip().lower()]

        if valid_df.empty:
            msg = "No valid coordinates found for selected filters."
            cls._draw_empty_message(ax, msg)
            fig.tight_layout(pad=1.2)
            return fig

        v_lat = pd.to_numeric(valid_df["Latitude"])
        v_lon = pd.to_numeric(valid_df["Longitude"])

        # Determine if zoomed to a single city or national view
        is_zoomed = bool(city_filter and str(city_filter).lower() != "all")

        # Map colors by severity
        plotted_handles = []
        for sev, color in [("Minor", cls.SEVERITY_COLORS["Minor"]),
                           ("Serious", cls.SEVERITY_COLORS["Serious"]),
                           ("Fatal", cls.SEVERITY_COLORS["Fatal"])]:
            sub = valid_df[valid_df["Severity"] == sev] if "Severity" in valid_df.columns else pd.DataFrame()
            if not sub.empty:
                s_lat = pd.to_numeric(sub["Latitude"])
                s_lon = pd.to_numeric(sub["Longitude"])
                scatter = ax.scatter(
                    s_lon, s_lat, c=color, label=f"{sev} ({len(sub)})",
                    alpha=0.70, edgecolors="white", linewidths=0.6,
                    s=55 if is_zoomed else 38, zorder=3
                )
                plotted_handles.append(scatter)

        # Plot other/unknown severities if present
        if "Severity" in valid_df.columns:
            other = valid_df[~valid_df["Severity"].isin(["Minor", "Serious", "Fatal"])]
            if not other.empty:
                ax.scatter(
                    pd.to_numeric(other["Longitude"]), pd.to_numeric(other["Latitude"]),
                    c="#94A3B8", label=f"Other ({len(other)})", alpha=0.55,
                    edgecolors="white", linewidths=0.5, s=30, zorder=3
                )

        # Annotate major municipal cluster centroids when in national view
        if not is_zoomed and "City" in valid_df.columns and len(valid_df) > 10:
            top_cities = valid_df["City"].dropna().astype(str).str.strip().value_counts().head(5)
            for city_name, count in top_cities.items():
                c_data = valid_df[valid_df["City"].astype(str).str.strip() == city_name]
                c_lat_mean = pd.to_numeric(c_data["Latitude"]).mean()
                c_lon_mean = pd.to_numeric(c_data["Longitude"]).mean()
                ax.annotate(
                    f"{city_name} ({count})",
                    xy=(c_lon_mean, c_lat_mean),
                    xytext=(c_lon_mean, c_lat_mean + (v_lat.max() - v_lat.min()) * 0.045),
                    fontsize=8, fontweight="bold", color=cls.THEME["navy"],
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color=cls.THEME["muted_text"], lw=0.8)
                )

        # Auto-compute tight zoom bounds with safe padding
        min_x, max_x = v_lon.min(), v_lon.max()
        min_y, max_y = v_lat.min(), v_lat.max()
        pad_x = max((max_x - min_x) * 0.025, 0.008)
        pad_y = max((max_y - min_y) * 0.025, 0.008)

        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Dynamic title based on scope
        if is_zoomed:
            title_text = f"Accident Hotspots: {city_filter} (Plotted: {len(valid_df)})"
        else:
            title_text = f"Geographic Accident Scatter Hotspots (N={len(valid_df)})"

        cls._apply_axes_styling(ax, title_text, xlabel="Longitude (Degrees W/E)", ylabel="Latitude (Degrees N/S)")

        # Legend with clean background card
        if plotted_handles or ax.get_legend_handles_labels()[0]:
            ax.legend(
                frameon=True, facecolor="#FFFFFF", edgecolor=cls.THEME["grid"],
                fontsize=8.5, loc="upper right", framealpha=0.92
            )

        # Ensure figure fits perfectly without clipping or overflowing
        fig.tight_layout(pad=0.5)
        return fig

    @staticmethod
    def save_figure(fig: Figure, filepath: str, dpi: int = 300) -> Tuple[bool, Optional[str]]:
        """Saves Matplotlib figure to image file."""
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            return True, None
        except Exception as e:
            return False, f"Failed to save chart: {str(e)}"
