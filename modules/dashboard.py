"""
Main Desktop Dashboard for Traffic Accident Analyzer.
Built with Tkinter, ttk, and embedded Matplotlib canvases.
Features a modern Dark Navy, Blue, and White aesthetic with sidebar navigation.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import pandas as pd
import numpy as np

from .data_loader import DataLoader
from .data_cleaning import DataCleaner
from .analysis import AccidentAnalyzer
from .visualizations import Visualizer
from .reports import ReportGenerator


class ModernDashboard(tk.Tk):
    """Primary GUI Application Window for Traffic Accident Analyzer."""

    # UI Theme Palette
    COLORS = {
        "sidebar_bg": "#0F172A",      # Deep dark navy
        "sidebar_hover": "#1E293B",   # Slate navy
        "sidebar_active": "#2563EB",  # Bright accent blue
        "sidebar_text": "#94A3B8",    # Slate muted
        "sidebar_title": "#FFFFFF",   # Crisp white
        "header_bg": "#FFFFFF",
        "header_border": "#E2E8F0",
        "main_bg": "#F1F5F9",        # Soft light grey background
        "card_bg": "#FFFFFF",
        "card_border": "#E2E8F0",
        "text_dark": "#0F172A",
        "text_muted": "#64748B",
        "primary": "#2563EB",
        "primary_hover": "#1D4ED8",
        "secondary": "#64748B",
        "fatal_red": "#EF4444",
        "serious_amber": "#F59E0B",
        "minor_green": "#10B981",
        "sky_blue": "#0284C7",
        "purple": "#6366F1",
    }

    FONTS = {
        "title": ("Segoe UI", 13, "bold"),
        "subtitle": ("Segoe UI", 9),
        "header": ("Segoe UI", 14, "bold"),
        "section": ("Segoe UI", 11, "bold"),
        "body": ("Segoe UI", 9),
        "body_bold": ("Segoe UI", 9, "bold"),
        "kpi_value": ("Segoe UI", 18, "bold"),
        "kpi_label": ("Segoe UI", 8, "bold"),
        "small": ("Segoe UI", 8),
    }

    def __init__(self):
        super().__init__()

        self.title("Traffic Accident Analyzer - Road Safety Analytics")
        self.geometry("1340x840")
        self.minsize(1120, 700)
        self.configure(bg=self.COLORS["main_bg"])

        # Core logic components
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.analyzer = AccidentAnalyzer()
        self.current_df: pd.DataFrame = pd.DataFrame()
        self.active_filters: Dict[str, str] = {
            "city": "All",
            "severity": "All",
            "weather": "All",
            "road_surface": "All",
            "search": ""
        }

        # Track navigation buttons and main views
        self.nav_buttons: Dict[str, tk.Button] = {}
        self.view_frames: Dict[str, ttk.Frame] = {}
        self.current_view: str = "dashboard"

        # Cached Matplotlib figures and canvases for smooth rendering
        self.dash_fig_sev = None
        self.dash_canvas_sev = None
        self.dash_fig_time = None
        self.dash_canvas_time = None

        self.analytics_fig = None
        self.analytics_canvas = None

        self.geo_fig = None
        self.geo_canvas = None

        # Build UI Architecture
        self._setup_styles()
        self._build_layout()

        # Load sample dataset automatically on startup if available
        self._auto_load_sample()

    # -------------------------------------------------------------
    # Styling and ttk configuration
    # -------------------------------------------------------------
    def _setup_styles(self):
        """Configures ttk style themes for professional widget presentation."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.COLORS["main_bg"], foreground=self.COLORS["text_dark"], font=self.FONTS["body"])
        
        # Treeview styling
        style.configure(
            "Custom.Treeview",
            background="#FFFFFF",
            foreground=self.COLORS["text_dark"],
            fieldbackground="#FFFFFF",
            rowheight=26,
            font=self.FONTS["body"],
            borderwidth=1,
            relief="flat"
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#F8FAFC",
            foreground=self.COLORS["text_dark"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padding=(6, 4)
        )
        style.map("Custom.Treeview.Heading", background=[("active", "#E2E8F0")])
        style.map("Custom.Treeview", background=[("selected", "#EFF6FF")], foreground=[("selected", "#1E3A8A")])

        # Combobox styling
        style.configure("Custom.TCombobox", padding=4, relief="flat", font=self.FONTS["body"])

    # -------------------------------------------------------------
    # Master Layout (Sidebar + Header + Content)
    # -------------------------------------------------------------
    def _build_layout(self):
        """Constructs sidebar, header, and multi-view content container."""
        # 1. Left Sidebar
        self.sidebar = tk.Frame(self, bg=self.COLORS["sidebar_bg"], width=290)
        self.sidebar.pack(side="left", fill="y", expand=False)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # 2. Main Right Container
        self.main_container = tk.Frame(self, bg=self.COLORS["main_bg"])
        self.main_container.pack(side="right", fill="both", expand=True)

        # 3. Top Header Bar
        self.header_bar = tk.Frame(self.main_container, bg=self.COLORS["header_bg"], height=58,
                                   highlightbackground=self.COLORS["header_border"], highlightthickness=1)
        self.header_bar.pack(side="top", fill="x")
        self.header_bar.pack_propagate(False)
        self._build_header()

        # 4. Content Area
        self.content_area = tk.Frame(self.main_container, bg=self.COLORS["main_bg"])
        self.content_area.pack(side="bottom", fill="both", expand=True, padx=16, pady=12)

        # 5. Initialize View Frames
        self._init_views()
        self.switch_view("dashboard")

    # -------------------------------------------------------------
    # Sidebar Construction
    # -------------------------------------------------------------
    def _build_sidebar(self):
        """Constructs sidebar branding, navigation buttons, and dataset status."""
        # Brand / Title Block
        brand_frame = tk.Frame(self.sidebar, bg=self.COLORS["sidebar_bg"], padx=16, pady=18)
        brand_frame.pack(fill="x")

        icon_title = tk.Label(
            brand_frame, text="Traffic Analyzer",
            bg=self.COLORS["sidebar_bg"], fg=self.COLORS["sidebar_title"],
            font=self.FONTS["title"], anchor="w"
        )
        icon_title.pack(fill="x")

        subtitle = tk.Label(
            brand_frame, text="Road Safety Data Analytics",
            bg=self.COLORS["sidebar_bg"], fg=self.COLORS["sidebar_text"],
            font=self.FONTS["subtitle"], anchor="w"
        )
        subtitle.pack(fill="x", pady=(2, 0))

        # Divider
        div = tk.Frame(self.sidebar, bg=self.COLORS["sidebar_hover"], height=1)
        div.pack(fill="x", padx=14, pady=4)

        # Navigation Section
        nav_container = tk.Frame(self.sidebar, bg=self.COLORS["sidebar_bg"], pady=8)
        nav_container.pack(fill="x")

        nav_items = [
            ("dashboard", "📊 Dashboard Overview"),
            ("explorer", "📁 Dataset Explorer"),
            ("cleaning", "🧹 Data Cleaning Studio"),
            ("analytics", "📈 In-Depth Analytics"),
            ("geo", "🌐 Geospatial Hotspots"),
            ("reports", "📑 Reports & Export"),
        ]

        for view_key, label_text in nav_items:
            btn = tk.Button(
                nav_container,
                text=label_text,
                anchor="w",
                padx=16,
                pady=9,
                font=self.FONTS["body_bold"],
                bg=self.COLORS["sidebar_bg"],
                fg=self.COLORS["sidebar_text"],
                activebackground=self.COLORS["sidebar_hover"],
                activeforeground="#FFFFFF",
                bd=0,
                cursor="hand2",
                command=lambda k=view_key: self.switch_view(k)
            )
            btn.pack(fill="x", pady=2)
            self.nav_buttons[view_key] = btn

        # Quick Actions Section
        action_frame = tk.Frame(self.sidebar, bg=self.COLORS["sidebar_bg"], padx=14, pady=16)
        action_frame.pack(side="bottom", fill="x")

        sample_btn = tk.Button(
            action_frame,
            text="⚡ Load Sample Dataset",
            bg=self.COLORS["sidebar_active"],
            fg="#FFFFFF",
            activebackground=self.COLORS["primary_hover"],
            activeforeground="#FFFFFF",
            font=self.FONTS["body_bold"],
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._load_sample_clicked
        )
        sample_btn.pack(fill="x", pady=(0, 10))

        # Dataset Status Card in Sidebar Footer
        self.sidebar_status_card = tk.Frame(
            action_frame,
            bg=self.COLORS["sidebar_hover"],
            padx=10,
            pady=8,
            highlightbackground="#334155",
            highlightthickness=1
        )
        self.sidebar_status_card.pack(fill="x")

        self.lbl_side_dataset = tk.Label(
            self.sidebar_status_card, text="Dataset: None loaded",
            bg=self.COLORS["sidebar_hover"], fg="#F8FAFC",
            font=("Segoe UI", 8, "bold"), anchor="w", wraplength=240
        )
        self.lbl_side_dataset.pack(fill="x")

        self.lbl_side_records = tk.Label(
            self.sidebar_status_card, text="0 records | 0 filters active",
            bg=self.COLORS["sidebar_hover"], fg=self.COLORS["sidebar_text"],
            font=self.FONTS["small"], anchor="w"
        )
        self.lbl_side_records.pack(fill="x", pady=(2, 0))

    # -------------------------------------------------------------
    # Top Header Bar Construction
    # -------------------------------------------------------------
    def _build_header(self):
        """Constructs breadcrumbs and top action toolbar."""
        left_box = tk.Frame(self.header_bar, bg=self.COLORS["header_bg"], padx=18)
        left_box.pack(side="left", fill="y")

        self.lbl_header_title = tk.Label(
            left_box, text="Dashboard Overview",
            bg=self.COLORS["header_bg"], fg=self.COLORS["text_dark"],
            font=self.FONTS["header"]
        )
        self.lbl_header_title.pack(side="left", pady=14)

        # Right Actions
        right_box = tk.Frame(self.header_bar, bg=self.COLORS["header_bg"], padx=16)
        right_box.pack(side="right", fill="y")

        btn_browse = tk.Button(
            right_box, text="📂 Open CSV / Excel...",
            bg="#F8FAFC", fg=self.COLORS["text_dark"],
            activebackground="#E2E8F0",
            font=self.FONTS["body_bold"],
            bd=1, relief="solid", padx=12, pady=5,
            cursor="hand2",
            command=self._browse_dataset
        )
        btn_browse.pack(side="left", padx=6, pady=12)

        btn_reset_filters = tk.Button(
            right_box, text="🔄 Reset Filters",
            bg="#F8FAFC", fg=self.COLORS["text_muted"],
            activebackground="#E2E8F0",
            font=self.FONTS["body"],
            bd=1, relief="solid", padx=10, pady=5,
            cursor="hand2",
            command=self.clear_all_filters
        )
        btn_reset_filters.pack(side="left", padx=4, pady=12)

    # -------------------------------------------------------------
    # View Switching Architecture
    # -------------------------------------------------------------
    def _init_views(self):
        """Creates container frames for each primary view."""
        self.view_frames["dashboard"] = self._create_dashboard_view()
        self.view_frames["explorer"] = self._create_explorer_view()
        self.view_frames["cleaning"] = self._create_cleaning_view()
        self.view_frames["analytics"] = self._create_analytics_view()
        self.view_frames["geo"] = self._create_geo_view()
        self.view_frames["reports"] = self._create_reports_view()

    def switch_view(self, view_key: str):
        """Toggles active view and highlights sidebar navigation."""
        self.current_view = view_key

        # Update sidebar button appearances
        for k, btn in self.nav_buttons.items():
            if k == view_key:
                btn.configure(bg=self.COLORS["sidebar_hover"], fg="#38BDF8")
            else:
                btn.configure(bg=self.COLORS["sidebar_bg"], fg=self.COLORS["sidebar_text"])

        # Hide all frames, show selected
        for k, frame in self.view_frames.items():
            frame.pack_forget()

        # Update title text
        titles = {
            "dashboard": "Dashboard Overview",
            "explorer": "Dataset Explorer & Filter",
            "cleaning": "Data Cleaning & Preprocessing Studio",
            "analytics": "In-Depth Accident Analytics",
            "geo": "Geospatial Accident Hotspots",
            "reports": "Reports & Analytics Export Center",
        }
        self.lbl_header_title.configure(text=titles.get(view_key, "Traffic Accident Analyzer"))

        if view_key in self.view_frames:
            self.view_frames[view_key].pack(fill="both", expand=True)

        # Refresh the active view's specific visualizations
        self._refresh_view(view_key)

    # -------------------------------------------------------------
    # VIEW 1: Dashboard Overview
    # -------------------------------------------------------------
    def _create_dashboard_view(self) -> ttk.Frame:
        """Constructs the high-level executive dashboard view."""
        frame = ttk.Frame(self.content_area)

        # Top Section: 6 KPI Cards in a grid
        kpi_container = tk.Frame(frame, bg=self.COLORS["main_bg"])
        kpi_container.pack(side="top", fill="x", pady=(0, 10))

        for i in range(6):
            kpi_container.columnconfigure(i, weight=1)

        self.kpi_widgets = {}
        card_configs = [
            ("total", "TOTAL ACCIDENTS", self.COLORS["primary"], "All recorded"),
            ("fatal", "FATAL ACCIDENTS", self.COLORS["fatal_red"], "Critical casualty"),
            ("serious", "SERIOUS ACCIDENTS", self.COLORS["serious_amber"], "Severe injury"),
            ("minor", "MINOR ACCIDENTS", self.COLORS["minor_green"], "Slight injury"),
            ("peak_hour", "PEAK ACCIDENT HOUR", self.COLORS["sky_blue"], "Highest frequency"),
            ("locations", "UNIQUE LOCATIONS", self.COLORS["purple"], "Cities recorded"),
        ]

        for idx, (key, label, accent_color, subtext) in enumerate(card_configs):
            card = tk.Frame(
                kpi_container, bg=self.COLORS["card_bg"],
                highlightbackground=self.COLORS["card_border"], highlightthickness=1,
                padx=12, pady=10
            )
            card.grid(row=0, column=idx, padx=4, sticky="nsew")

            # Top accent line
            top_bar = tk.Frame(card, bg=accent_color, height=3)
            top_bar.pack(fill="x", pady=(0, 6))

            lbl_title = tk.Label(card, text=label, bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"], font=self.FONTS["kpi_label"])
            lbl_title.pack(anchor="w")

            lbl_val = tk.Label(card, text="0", bg=self.COLORS["card_bg"], fg=self.COLORS["text_dark"], font=self.FONTS["kpi_value"])
            lbl_val.pack(anchor="w", pady=(2, 2))

            lbl_sub = tk.Label(card, text=subtext, bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"], font=self.FONTS["small"])
            lbl_sub.pack(anchor="w")

            self.kpi_widgets[key] = (lbl_val, lbl_sub)

        # Middle Section: Quick Filter Bar
        dash_filter_frame = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=12, pady=8
        )
        dash_filter_frame.pack(side="top", fill="x", pady=(0, 10))

        tk.Label(dash_filter_frame, text="Filter By:", bg=self.COLORS["card_bg"], font=self.FONTS["body_bold"]).pack(side="left", padx=(0, 8))

        # City Filter
        tk.Label(dash_filter_frame, text="City:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(4, 2))
        self.dash_cb_city = ttk.Combobox(dash_filter_frame, state="readonly", width=14)
        self.dash_cb_city.pack(side="left", padx=(0, 8))
        self.dash_cb_city.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("city", self.dash_cb_city.get()))

        # Severity Filter
        tk.Label(dash_filter_frame, text="Severity:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(4, 2))
        self.dash_cb_sev = ttk.Combobox(dash_filter_frame, values=["All", "Fatal", "Serious", "Minor"], state="readonly", width=10)
        self.dash_cb_sev.set("All")
        self.dash_cb_sev.pack(side="left", padx=(0, 8))
        self.dash_cb_sev.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("severity", self.dash_cb_sev.get()))

        # Weather Filter
        tk.Label(dash_filter_frame, text="Weather:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(4, 2))
        self.dash_cb_weather = ttk.Combobox(dash_filter_frame, state="readonly", width=12)
        self.dash_cb_weather.pack(side="left", padx=(0, 8))
        self.dash_cb_weather.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("weather", self.dash_cb_weather.get()))

        # Reset button
        btn_dash_clear = tk.Button(
            dash_filter_frame, text="Clear Filters",
            bg="#F8FAFC", fg=self.COLORS["text_muted"],
            activebackground="#E2E8F0",
            font=self.FONTS["small"],
            bd=1, relief="solid", padx=8, pady=2,
            command=self.clear_all_filters
        )
        btn_dash_clear.pack(side="left", padx=(6, 0))

        # Bottom Section: Two embedded charts side-by-side
        chart_split_frame = tk.Frame(frame, bg=self.COLORS["main_bg"])
        chart_split_frame.pack(side="bottom", fill="both", expand=True)
        chart_split_frame.columnconfigure(0, weight=1)
        chart_split_frame.columnconfigure(1, weight=1)
        chart_split_frame.rowconfigure(0, weight=1)

        # Left Chart: Severity Donut
        card_sev = tk.Frame(chart_split_frame, bg=self.COLORS["card_bg"],
                            highlightbackground=self.COLORS["card_border"], highlightthickness=1, padx=6, pady=6)
        card_sev.grid(row=0, column=0, padx=(0, 5), sticky="nsew")

        bar_sev_top = tk.Frame(card_sev, bg=self.COLORS["card_bg"])
        bar_sev_top.pack(fill="x", pady=(2, 4))
        tk.Label(bar_sev_top, text="Severity Proportion", bg=self.COLORS["card_bg"], font=self.FONTS["section"]).pack(side="left", padx=4)
        btn_save_sev = tk.Button(
            bar_sev_top, text="💾 Save", bg="#F8FAFC", font=self.FONTS["small"], bd=1, relief="solid",
            command=lambda: self._save_chart_as_image(self.dash_fig_sev, "severity_chart.png")
        )
        btn_save_sev.pack(side="right", padx=4)

        self.dash_sev_container = tk.Frame(card_sev, bg=self.COLORS["card_bg"])
        self.dash_sev_container.pack(fill="both", expand=True)

        # Right Chart: Hourly Line or Geographic Hotspots
        card_time = tk.Frame(chart_split_frame, bg=self.COLORS["card_bg"],
                             highlightbackground=self.COLORS["card_border"], highlightthickness=1, padx=6, pady=6)
        card_time.grid(row=0, column=1, padx=(5, 0), sticky="nsew")

        bar_time_top = tk.Frame(card_time, bg=self.COLORS["card_bg"])
        bar_time_top.pack(fill="x", pady=(2, 4))
        self.lbl_dash_right_title = tk.Label(bar_time_top, text="24-Hour Accident Timeline", bg=self.COLORS["card_bg"], font=self.FONTS["section"])
        self.lbl_dash_right_title.pack(side="left", padx=4)

        self.dash_right_mode = "timeline"

        btn_save_time = tk.Button(
            bar_time_top, text="💾 Save", bg="#F8FAFC", font=self.FONTS["small"], bd=1, relief="solid",
            command=lambda: self._save_chart_as_image(self.dash_fig_time, f"dashboard_{self.dash_right_mode}.png")
        )
        btn_save_time.pack(side="right", padx=4)

        self.btn_dash_geo_toggle = tk.Button(
            bar_time_top, text="Map", bg="#F1F5F9", fg=self.COLORS["text_dark"],
            font=self.FONTS["small"], bd=1, relief="solid", padx=6, pady=1, cursor="hand2",
            command=lambda: self._toggle_dash_right_chart("geo")
        )
        self.btn_dash_geo_toggle.pack(side="right", padx=2)

        self.btn_dash_time_toggle = tk.Button(
            bar_time_top, text="⏱️ Timeline", bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["small"], bd=0, padx=6, pady=1, cursor="hand2",
            command=lambda: self._toggle_dash_right_chart("timeline")
        )
        self.btn_dash_time_toggle.pack(side="right", padx=2)

        self.dash_time_container = tk.Frame(card_time, bg=self.COLORS["card_bg"])
        self.dash_time_container.pack(fill="both", expand=True)

        return frame

    # -------------------------------------------------------------
    # VIEW 2: Dataset Explorer
    # -------------------------------------------------------------
    def _create_explorer_view(self) -> ttk.Frame:
        """Constructs data grid explorer with search, sorting, filtering and CSV export."""
        frame = ttk.Frame(self.content_area)

        # Top Control Bar (Search & Filter)
        filter_bar = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=12, pady=10
        )
        filter_bar.pack(side="top", fill="x", pady=(0, 10))

        # Search Box
        tk.Label(filter_bar, text="🔍 Search:", bg=self.COLORS["card_bg"], font=self.FONTS["body_bold"]).pack(side="left", padx=(0, 4))
        self.ent_explorer_search = ttk.Entry(filter_bar, width=18)
        self.ent_explorer_search.pack(side="left", padx=(0, 10))
        self.ent_explorer_search.bind("<Return>", lambda e: self._on_search_submitted())

        btn_search = tk.Button(
            filter_bar, text="Search", bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["small"], bd=0, padx=8, pady=3,
            command=self._on_search_submitted
        )
        btn_search.pack(side="left", padx=(0, 12))

        # City Dropdown
        tk.Label(filter_bar, text="City:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(0, 2))
        self.exp_cb_city = ttk.Combobox(filter_bar, state="readonly", width=12)
        self.exp_cb_city.pack(side="left", padx=(0, 8))
        self.exp_cb_city.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("city", self.exp_cb_city.get()))

        # Severity Dropdown
        tk.Label(filter_bar, text="Severity:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(0, 2))
        self.exp_cb_sev = ttk.Combobox(filter_bar, values=["All", "Fatal", "Serious", "Minor"], state="readonly", width=10)
        self.exp_cb_sev.set("All")
        self.exp_cb_sev.pack(side="left", padx=(0, 8))
        self.exp_cb_sev.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("severity", self.exp_cb_sev.get()))

        # Weather Dropdown
        tk.Label(filter_bar, text="Weather:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(0, 2))
        self.exp_cb_weather = ttk.Combobox(filter_bar, state="readonly", width=12)
        self.exp_cb_weather.pack(side="left", padx=(0, 8))
        self.exp_cb_weather.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("weather", self.exp_cb_weather.get()))

        # Road Surface Dropdown
        tk.Label(filter_bar, text="Road Surface:", bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"]).pack(side="left", padx=(0, 2))
        self.exp_cb_surface = ttk.Combobox(filter_bar, state="readonly", width=12)
        self.exp_cb_surface.pack(side="left", padx=(0, 8))
        self.exp_cb_surface.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed("road_surface", self.exp_cb_surface.get()))

        # Clear button
        btn_exp_clear = tk.Button(
            filter_bar, text="Clear Filters", bg="#F8FAFC", font=self.FONTS["small"],
            bd=1, relief="solid", padx=8, pady=3,
            command=self.clear_all_filters
        )
        btn_exp_clear.pack(side="left", padx=(4, 8))

        # Export filtered button on right
        btn_exp_csv = tk.Button(
            filter_bar, text="📥 Export Filtered to CSV",
            bg=self.COLORS["minor_green"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=10, pady=4,
            command=self._export_filtered_dataset
        )
        btn_exp_csv.pack(side="right")

        # Table Container
        table_card = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1
        )
        table_card.pack(fill="both", expand=True)

        # Treeview + Scrollbars
        scroll_y = ttk.Scrollbar(table_card, orient="vertical")
        scroll_x = ttk.Scrollbar(table_card, orient="horizontal")

        self.tree_explorer = ttk.Treeview(
            table_card,
            style="Custom.Treeview",
            selectmode="browse",
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        scroll_y.config(command=self.tree_explorer.yview)
        scroll_x.config(command=self.tree_explorer.xview)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.tree_explorer.pack(side="left", fill="both", expand=True)

        # Table Footer Bar
        table_footer = tk.Frame(frame, bg=self.COLORS["main_bg"], pady=6)
        table_footer.pack(side="bottom", fill="x")

        self.lbl_explorer_status = tk.Label(
            table_footer, text="Showing 0 of 0 records",
            bg=self.COLORS["main_bg"], fg=self.COLORS["text_muted"], font=self.FONTS["body"]
        )
        self.lbl_explorer_status.pack(side="left")

        self.lbl_explorer_meta = tk.Label(
            table_footer, text="Columns: 0",
            bg=self.COLORS["main_bg"], fg=self.COLORS["text_muted"], font=self.FONTS["small"]
        )
        self.lbl_explorer_meta.pack(side="right")

        return frame

    # -------------------------------------------------------------
    # VIEW 3: Data Cleaning Studio
    # -------------------------------------------------------------
    def _create_cleaning_view(self) -> ttk.Frame:
        """Constructs data audit, duplicate removal, imputation, and normalization interface."""
        frame = ttk.Frame(self.content_area)

        # Top Quality Metrics Summary
        top_metrics = tk.Frame(frame, bg=self.COLORS["main_bg"])
        top_metrics.pack(side="top", fill="x", pady=(0, 10))
        for i in range(4):
            top_metrics.columnconfigure(i, weight=1)

        self.clean_metric_labels = {}
        m_configs = [
            ("total_rows", "Total Loaded Rows", self.COLORS["primary"]),
            ("duplicates", "Duplicate Records", self.COLORS["serious_amber"]),
            ("missing_cols", "Columns with Missing Data", self.COLORS["fatal_red"]),
            ("invalid_coords", "Invalid / Zero Coordinates", self.COLORS["purple"])
        ]
        for idx, (key, title, color) in enumerate(m_configs):
            card = tk.Frame(top_metrics, bg=self.COLORS["card_bg"],
                            highlightbackground=self.COLORS["card_border"], highlightthickness=1, padx=12, pady=10)
            card.grid(row=0, column=idx, padx=4, sticky="nsew")

            bar = tk.Frame(card, bg=color, height=3)
            bar.pack(fill="x", pady=(0, 4))

            tk.Label(card, text=title, bg=self.COLORS["card_bg"], fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(anchor="w")
            lbl_v = tk.Label(card, text="0", bg=self.COLORS["card_bg"], fg=self.COLORS["text_dark"], font=self.FONTS["kpi_value"])
            lbl_v.pack(anchor="w")
            self.clean_metric_labels[key] = lbl_v

        # Middle Split: Left = Column Quality Table, Right = Cleaning Operations & Audit Log
        split_frame = tk.Frame(frame, bg=self.COLORS["main_bg"])
        split_frame.pack(side="bottom", fill="both", expand=True)
        split_frame.columnconfigure(0, weight=3)
        split_frame.columnconfigure(1, weight=2)
        split_frame.rowconfigure(0, weight=1)

        # Left: Column Missingness Table
        left_card = tk.Frame(split_frame, bg=self.COLORS["card_bg"],
                             highlightbackground=self.COLORS["card_border"], highlightthickness=1, padx=10, pady=10)
        left_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        tk.Label(left_card, text="Column Quality & Completeness Audit", bg=self.COLORS["card_bg"], font=self.FONTS["section"]).pack(anchor="w", pady=(0, 8))

        scroll_qual = ttk.Scrollbar(left_card, orient="vertical")
        self.tree_quality = ttk.Treeview(
            left_card,
            columns=("Column", "Data Type", "Missing Count", "Missing %", "Valid Records"),
            show="headings",
            style="Custom.Treeview",
            yscrollcommand=scroll_qual.set
        )
        scroll_qual.config(command=self.tree_quality.yview)
        scroll_qual.pack(side="right", fill="y")
        self.tree_quality.pack(side="left", fill="both", expand=True)

        for col, width in [("Column", 130), ("Data Type", 80), ("Missing Count", 90), ("Missing %", 80), ("Valid Records", 90)]:
            self.tree_quality.heading(col, text=col)
            self.tree_quality.column(col, width=width, anchor="center" if "Count" in col or "%" in col else "w")

        # Right: Cleaning Controls & Audit Log
        right_card = tk.Frame(split_frame, bg=self.COLORS["card_bg"],
                              highlightbackground=self.COLORS["card_border"], highlightthickness=1, padx=12, pady=10)
        right_card.grid(row=0, column=1, padx=(6, 0), sticky="nsew")

        tk.Label(right_card, text="Automated Cleaning Options", bg=self.COLORS["card_bg"], font=self.FONTS["section"]).pack(anchor="w", pady=(0, 8))

        # Checkboxes for cleaning operations
        self.var_drop_dups = tk.BooleanVar(value=True)
        self.var_norm_sev = tk.BooleanVar(value=True)
        self.var_fill_cat = tk.BooleanVar(value=True)
        self.var_fill_num = tk.BooleanVar(value=True)
        self.var_sanitize_coords = tk.BooleanVar(value=True)

        cb1 = tk.Checkbutton(right_card, text="Detect and remove duplicate records",
                             variable=self.var_drop_dups, bg=self.COLORS["card_bg"], font=self.FONTS["body"])
        cb1.pack(anchor="w", pady=2)

        cb2 = tk.Checkbutton(right_card, text="Normalize Severity labels (Fatal, Serious, Minor)",
                             variable=self.var_norm_sev, bg=self.COLORS["card_bg"], font=self.FONTS["body"])
        cb2.pack(anchor="w", pady=2)

        cb3 = tk.Checkbutton(right_card, text="Impute missing categorical fields with 'Unknown'",
                             variable=self.var_fill_cat, bg=self.COLORS["card_bg"], font=self.FONTS["body"])
        cb3.pack(anchor="w", pady=2)

        cb4 = tk.Checkbutton(right_card, text="Impute missing numeric fields with column median",
                             variable=self.var_fill_num, bg=self.COLORS["card_bg"], font=self.FONTS["body"])
        cb4.pack(anchor="w", pady=2)

        cb5 = tk.Checkbutton(right_card, text="Sanitize out-of-bounds coordinates (set to NaN)",
                             variable=self.var_sanitize_coords, bg=self.COLORS["card_bg"], font=self.FONTS["body"])
        cb5.pack(anchor="w", pady=2)

        # Action Buttons
        btn_box = tk.Frame(right_card, bg=self.COLORS["card_bg"], pady=8)
        btn_box.pack(fill="x")

        btn_run_clean = tk.Button(
            btn_box, text="⚡ Run Cleaning Pipeline",
            bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=12, pady=6,
            command=self._execute_cleaning
        )
        btn_run_clean.pack(side="left", padx=(0, 6))

        btn_restore = tk.Button(
            btn_box, text="↩️ Restore Original",
            bg="#F8FAFC", fg=self.COLORS["text_dark"],
            font=self.FONTS["body"], bd=1, relief="solid", padx=10, pady=5,
            command=self._restore_original_data
        )
        btn_restore.pack(side="left", padx=4)

        btn_export_clean = tk.Button(
            btn_box, text="📥 Export Cleaned CSV",
            bg=self.COLORS["minor_green"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=10, pady=5,
            command=self._export_cleaned_dataset
        )
        btn_export_clean.pack(side="right")

        # Audit Log Box
        tk.Label(right_card, text="Cleaning Execution & Audit Log:", bg=self.COLORS["card_bg"], font=self.FONTS["body_bold"]).pack(anchor="w", pady=(8, 4))
        
        log_scroll = ttk.Scrollbar(right_card, orient="vertical")
        self.txt_clean_log = tk.Text(
            right_card, height=10, bg="#F8FAFC", fg=self.COLORS["text_dark"],
            font=("Consolas", 8), yscrollcommand=log_scroll.set, bd=1, relief="solid"
        )
        log_scroll.config(command=self.txt_clean_log.yview)
        log_scroll.pack(side="right", fill="y")
        self.txt_clean_log.pack(fill="both", expand=True)

        return frame

    # -------------------------------------------------------------
    # VIEW 4: In-Depth Analytics
    # -------------------------------------------------------------
    def _create_analytics_view(self) -> ttk.Frame:
        """Constructs tabbed / selectable deep-dive charts with save capability."""
        frame = ttk.Frame(self.content_area)

        # Top Chart Switcher Bar
        toolbar = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=12, pady=8
        )
        toolbar.pack(side="top", fill="x", pady=(0, 10))

        tk.Label(toolbar, text="Select Analysis Chart:", bg=self.COLORS["card_bg"], font=self.FONTS["body_bold"]).pack(side="left", padx=(0, 8))

        self.analytics_chart_types = [
            ("sev_bar", "Severity Bar Chart"),
            ("sev_pie", "Severity Donut Chart"),
            ("time_hour", "Hourly Accident Timeline"),
            ("time_month", "Monthly Frequency Chart"),
            ("locations", "Top 10 Locations Ranking"),
            ("weather", "Weather Condition Analysis"),
            ("surface", "Road Surface Conditions"),
            ("vehicles", "Vehicles Involved Distribution"),
            ("heatmap", "Pearson Correlation Heatmap"),
        ]

        self.cb_analytics_type = ttk.Combobox(
            toolbar,
            values=[title for _, title in self.analytics_chart_types],
            state="readonly",
            width=32,
            font=self.FONTS["body"]
        )
        self.cb_analytics_type.current(0)
        self.cb_analytics_type.pack(side="left", padx=(0, 12))
        self.cb_analytics_type.bind("<<ComboboxSelected>>", lambda e: self._render_analytics_chart())

        btn_save = tk.Button(
            toolbar, text="💾 Save Chart as PNG (300 DPI)",
            bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=12, pady=5,
            command=lambda: self._save_chart_as_image(self.analytics_fig, "accident_analysis_chart.png")
        )
        btn_save.pack(side="right")

        # Main Chart Display Canvas Card
        self.analytics_card = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=10, pady=10
        )
        self.analytics_card.pack(fill="both", expand=True)

        return frame

    # -------------------------------------------------------------
    # VIEW 5: Geospatial Hotspots
    # -------------------------------------------------------------
    def _create_geo_view(self) -> ttk.Frame:
        """Constructs geographic coordinate scatter viewer with an aside bar and proper UI."""
        frame = ttk.Frame(self.content_area)

        # Top Control / Info Bar
        info_bar = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=14, pady=8
        )
        info_bar.pack(side="top", fill="x", pady=(0, 4))

        tk.Label(
            info_bar,
            text="Geospatial Accident Scatter & Spatial Hotspots",
            bg=self.COLORS["card_bg"], fg=self.COLORS["text_dark"], font=self.FONTS["section"]
        ).pack(side="left")

        self.lbl_geo_header_scope = tk.Label(
            info_bar, text="Display: All Municipalities",
            bg="#EFF6FF", fg=self.COLORS["primary"], font=self.FONTS["small"],
            padx=8, pady=2, relief="solid", bd=1
        )
        self.lbl_geo_header_scope.pack(side="left", padx=12)

        btn_save_geo = tk.Button(
            info_bar, text="💾 Save Map (300 DPI)",
            bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=10, pady=4, cursor="hand2",
            command=lambda: self._save_chart_as_image(self.geo_fig, "accident_hotspots_map.png")
        )
        btn_save_geo.pack(side="right")

        btn_reset_geo_top = tk.Button(
            info_bar, text="🔄 Reset Zoom",
            bg="#F8FAFC", fg=self.COLORS["text_dark"],
            font=self.FONTS["small"], bd=1, relief="solid", padx=8, pady=3, cursor="hand2",
            command=self._reset_geo_view
        )
        btn_reset_geo_top.pack(side="right", padx=6)

        # Main Split Body: Left = Aside Bar, Right = Canvas Container
        geo_body = tk.Frame(frame, bg=self.COLORS["main_bg"])
        geo_body.pack(side="bottom", fill="both", expand=True)

        # --- LEFT ASIDE BAR (width=315, fits cleanly without overflow) ---
        self.geo_aside = tk.Frame(
            geo_body, bg=self.COLORS["card_bg"], width=340,
            highlightbackground=self.COLORS["card_border"], highlightthickness=1
        )
        self.geo_aside.pack(side="left", fill="y", padx=(0, 4))
        self.geo_aside.pack_propagate(False)

        # Scrollable container inside the aside bar to guarantee NO OVERFLOW
        aside_canvas = tk.Canvas(self.geo_aside, bg=self.COLORS["card_bg"], highlightthickness=0, bd=0)
        aside_scroll = ttk.Scrollbar(self.geo_aside, orient="vertical", command=aside_canvas.yview)
        self.geo_aside_content = tk.Frame(aside_canvas, bg=self.COLORS["card_bg"], padx=10, pady=8)

        self.geo_aside_content.bind(
            "<Configure>",
            lambda e: aside_canvas.configure(scrollregion=aside_canvas.bbox("all"))
        )
        aside_win = aside_canvas.create_window((0, 0), window=self.geo_aside_content, anchor="nw", width=320)

        def _on_aside_resize(event):
            aside_canvas.itemconfig(aside_win, width=event.width)
        self.geo_aside.bind("<Configure>", _on_aside_resize)

        aside_canvas.configure(yscrollcommand=aside_scroll.set)
        aside_scroll.pack(side="right", fill="y")
        aside_canvas.pack(side="left", fill="both", expand=True)

        # Card 1: 📍 Scope & Location Filter
        card_scope = tk.Frame(self.geo_aside_content, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=8)
        card_scope.pack(fill="x", pady=(0, 8))

        tk.Label(card_scope, text="📍 Scope & Location Focus", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["body_bold"]).pack(anchor="w", pady=(0, 4))
        
        tk.Label(card_scope, text="Municipality / City:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(anchor="w")
        self.geo_cb_city = ttk.Combobox(card_scope, state="readonly", font=self.FONTS["body"])
        self.geo_cb_city.pack(fill="x", pady=(2, 6))
        self.geo_cb_city.bind("<<ComboboxSelected>>", self._on_geo_city_changed)

        tk.Label(card_scope, text="Severity Filter:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(anchor="w")
        self.geo_cb_sev = ttk.Combobox(card_scope, values=["All Severities", "Fatal", "Serious", "Minor"], state="readonly", font=self.FONTS["body"])
        self.geo_cb_sev.set("All Severities")
        self.geo_cb_sev.pack(fill="x", pady=(2, 2))
        self.geo_cb_sev.bind("<<ComboboxSelected>>", self._on_geo_sev_changed)

        # Card 2: 📊 Coordinate Health & Bounding Box
        card_metrics = tk.Frame(self.geo_aside_content, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=8)
        card_metrics.pack(fill="x", pady=(0, 8))

        tk.Label(card_metrics, text="📊 Spatial Coordinate Metrics", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["body_bold"]).pack(anchor="w", pady=(0, 4))
        
        row_gps = tk.Frame(card_metrics, bg="#F8FAFC")
        row_gps.pack(fill="x", pady=1)
        tk.Label(row_gps, text="Plotted GPS:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(side="left")
        self.lbl_geo_valid_pts = tk.Label(row_gps, text="0 / 0 (0%)", bg="#F8FAFC", fg=self.COLORS["primary"], font=self.FONTS["body_bold"])
        self.lbl_geo_valid_pts.pack(side="right")

        row_lat = tk.Frame(card_metrics, bg="#F8FAFC")
        row_lat.pack(fill="x", pady=1)
        tk.Label(row_lat, text="Latitude Bounds:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(side="left")
        self.lbl_geo_lat_bounds = tk.Label(row_lat, text="[0° to 0° N]", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["small"])
        self.lbl_geo_lat_bounds.pack(side="right")

        row_lon = tk.Frame(card_metrics, bg="#F8FAFC")
        row_lon.pack(fill="x", pady=1)
        tk.Label(row_lon, text="Longitude Bounds:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]).pack(side="left")
        self.lbl_geo_lon_bounds = tk.Label(row_lon, text="[0° to 0° W]", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["small"])
        self.lbl_geo_lon_bounds.pack(side="right")

        row_cen = tk.Frame(card_metrics, bg="#F8FAFC")
        row_cen.pack(fill="x", pady=(3, 1))
        self.lbl_geo_centroid = tk.Label(row_cen, text="Centroid: (0.000°, 0.000°)", bg="#F8FAFC", fg=self.COLORS["secondary"], font=("Segoe UI", 8, "italic"))
        self.lbl_geo_centroid.pack(anchor="w")

        # Card 3: 🏷️ Severity Distribution (Pill Badges)
        card_sev_dist = tk.Frame(self.geo_aside_content, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=8)
        card_sev_dist.pack(fill="x", pady=(0, 8))

        tk.Label(card_sev_dist, text="🏷️ Severity Breakdown", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["body_bold"]).pack(anchor="w", pady=(0, 4))

        # Fatal pill
        pill_fatal = tk.Frame(card_sev_dist, bg="#FEF2F2", highlightbackground="#FCA5A5", highlightthickness=1, padx=6, pady=3)
        pill_fatal.pack(fill="x", pady=2)
        tk.Label(pill_fatal, text="● Fatal", bg="#FEF2F2", fg=self.COLORS["fatal_red"], font=self.FONTS["body_bold"]).pack(side="left")
        self.lbl_geo_sev_fatal = tk.Label(pill_fatal, text="0 (0%)", bg="#FEF2F2", fg=self.COLORS["fatal_red"], font=self.FONTS["small"])
        self.lbl_geo_sev_fatal.pack(side="right")

        # Serious pill
        pill_serious = tk.Frame(card_sev_dist, bg="#FFFBEB", highlightbackground="#FDE68A", highlightthickness=1, padx=6, pady=3)
        pill_serious.pack(fill="x", pady=2)
        tk.Label(pill_serious, text="● Serious", bg="#FFFBEB", fg=self.COLORS["serious_amber"], font=self.FONTS["body_bold"]).pack(side="left")
        self.lbl_geo_sev_serious = tk.Label(pill_serious, text="0 (0%)", bg="#FFFBEB", fg=self.COLORS["serious_amber"], font=self.FONTS["small"])
        self.lbl_geo_sev_serious.pack(side="right")

        # Minor pill
        pill_minor = tk.Frame(card_sev_dist, bg="#ECFDF5", highlightbackground="#A7F3D0", highlightthickness=1, padx=6, pady=3)
        pill_minor.pack(fill="x", pady=2)
        tk.Label(pill_minor, text="● Minor", bg="#ECFDF5", fg=self.COLORS["minor_green"], font=self.FONTS["body_bold"]).pack(side="left")
        self.lbl_geo_sev_minor = tk.Label(pill_minor, text="0 (0%)", bg="#ECFDF5", fg=self.COLORS["minor_green"], font=self.FONTS["small"])
        self.lbl_geo_sev_minor.pack(side="right")

        # Card 4: 🏙️ Top Hotspot Clusters
        card_clusters = tk.Frame(self.geo_aside_content, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=8)
        card_clusters.pack(fill="x", pady=(0, 8))

        tk.Label(card_clusters, text="🏙️ Top Municipal Clusters", bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["body_bold"]).pack(anchor="w", pady=(0, 2))
        tk.Label(card_clusters, text="Click to center and zoom map:", bg="#F8FAFC", fg=self.COLORS["text_muted"], font=("Segoe UI", 7, "italic")).pack(anchor="w", pady=(0, 4))

        self.geo_clusters_list_frame = tk.Frame(card_clusters, bg="#F8FAFC")
        self.geo_clusters_list_frame.pack(fill="x")

        # Card 5: Actions
        card_actions = tk.Frame(self.geo_aside_content, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=8)
        card_actions.pack(fill="x", pady=(0, 4))

        btn_reset_zoom = tk.Button(
            card_actions, text="🔄 Reset to National View",
            bg="#FFFFFF", fg=self.COLORS["text_dark"],
            font=self.FONTS["small"], bd=1, relief="solid", padx=8, pady=4, cursor="hand2",
            command=self._reset_geo_view
        )
        btn_reset_zoom.pack(fill="x", pady=2)

        btn_export_geo = tk.Button(
            card_actions, text="📥 Export Plotted GPS (CSV)",
            bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["small"], bd=0, padx=8, pady=4, cursor="hand2",
            command=self._export_geo_csv
        )
        btn_export_geo.pack(fill="x", pady=2)

        # --- RIGHT MAIN AREA: MAP CANVAS CONTAINER (fits without overflow) ---
        self.geo_card = tk.Frame(
            geo_body, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=0, pady=0
        )
        self.geo_card.pack(side="right", fill="both", expand=True)

        return frame

    def _create_reports_view(self) -> ttk.Frame:
        """Constructs full text report preview and multi-table CSV exporter."""
        frame = ttk.Frame(self.content_area)

        # Top Action Bar
        action_bar = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=12, pady=10
        )
        action_bar.pack(side="top", fill="x", pady=(0, 10))

        btn_gen = tk.Button(
            action_bar, text="🔄 Refresh Report",
            bg=self.COLORS["primary"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=12, pady=6,
            command=self._generate_and_display_report
        )
        btn_gen.pack(side="left", padx=(0, 8))

        btn_save_txt = tk.Button(
            action_bar, text="📄 Export Report (.txt)",
            bg="#F8FAFC", fg=self.COLORS["text_dark"],
            font=self.FONTS["body_bold"], bd=1, relief="solid", padx=10, pady=5,
            command=self._export_report_txt
        )
        btn_save_txt.pack(side="left", padx=(0, 8))

        btn_save_tables = tk.Button(
            action_bar, text="📊 Export Summary Tables (CSV)",
            bg=self.COLORS["minor_green"], fg="#FFFFFF",
            font=self.FONTS["body_bold"], bd=0, padx=12, pady=6,
            command=self._export_summary_tables
        )
        btn_save_tables.pack(side="left")

        # Report Preview Text Box
        report_card = tk.Frame(
            frame, bg=self.COLORS["card_bg"],
            highlightbackground=self.COLORS["card_border"], highlightthickness=1,
            padx=10, pady=10
        )
        report_card.pack(fill="both", expand=True)

        scroll_rep = ttk.Scrollbar(report_card, orient="vertical")
        self.txt_report_preview = tk.Text(
            report_card, bg="#FFFFFF", fg=self.COLORS["text_dark"],
            font=("Consolas", 9), yscrollcommand=scroll_rep.set, bd=0
        )
        scroll_rep.config(command=self.txt_report_preview.yview)
        scroll_rep.pack(side="right", fill="y")
        self.txt_report_preview.pack(fill="both", expand=True)

        return frame


    # -------------------------------------------------------------
    # Data Loading and Sample Handling
    # -------------------------------------------------------------
    def _auto_load_sample(self):
        """Attempts to load data/sample_accidents.csv on startup."""
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_accidents.csv")
        if not os.path.exists(sample_path):
            # Check relative to working directory
            sample_path = os.path.join("TrafficAccidentAnalyzer", "data", "sample_accidents.csv")
            if not os.path.exists(sample_path):
                sample_path = os.path.join("data", "sample_accidents.csv")

        if os.path.exists(sample_path):
            self._load_dataset_file(sample_path)

    def _load_sample_clicked(self):
        """Callback for Load Sample Dataset button."""
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_accidents.csv")
        if not os.path.exists(sample_path):
            sample_path = os.path.join("TrafficAccidentAnalyzer", "data", "sample_accidents.csv")
            if not os.path.exists(sample_path):
                sample_path = os.path.join("data", "sample_accidents.csv")

        if not os.path.exists(sample_path):
            messagebox.showerror("Sample Not Found", f"Could not locate sample_accidents.csv at: {sample_path}")
            return

        self._load_dataset_file(sample_path)
        messagebox.showinfo("Sample Loaded", "Successfully loaded benchmark sample accident dataset!")

    def _browse_dataset(self):
        """Opens file dialog for user to select CSV or Excel file."""
        file_path = filedialog.askopenfilename(
            title="Open Accident Dataset",
            filetypes=[
                ("Supported Files", "*.csv *.xlsx *.xls"),
                ("CSV Files (*.csv)", "*.csv"),
                ("Excel Files (*.xlsx *.xls)", "*.xlsx *.xls"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self._load_dataset_file(file_path)

    def _load_dataset_file(self, file_path: str):
        """Loads file via DataLoader, updates cleaner, analyzer, dropdowns, and UI."""
        df, err = self.loader.load_file(file_path)
        if err or df is None:
            messagebox.showerror("Import Error", f"Unable to load dataset:\n{err}")
            return

        self.current_df = df.copy()
        self.cleaner.set_dataframe(df)
        self.analyzer.set_data(df)

        # Clear filters and populate dropdown values
        self._populate_dropdowns(df)
        self._update_all_views()

        # Update sidebar status badge
        fname = os.path.basename(file_path)
        self.lbl_side_dataset.configure(text=f"Dataset: {fname}")
        self.lbl_side_records.configure(text=f"{len(df):,} records | 0 filters")

    def _populate_dropdowns(self, df: pd.DataFrame):
        """Extracts unique values for city, weather, road surface filters."""
        # Cities
        if "City" in df.columns:
            cities = ["All"] + sorted([str(x) for x in df["City"].dropna().unique() if str(x).strip()])
            self.dash_cb_city["values"] = cities
            self.dash_cb_city.set("All")
            self.exp_cb_city["values"] = cities
            self.exp_cb_city.set("All")
        else:
            self.dash_cb_city["values"] = ["All (N/A)"]
            self.dash_cb_city.set("All (N/A)")
            self.exp_cb_city["values"] = ["All (N/A)"]
            self.exp_cb_city.set("All (N/A)")

        # Weather
        if "Weather" in df.columns:
            weathers = ["All"] + sorted([str(x).title() for x in df["Weather"].dropna().unique() if str(x).strip()])
            self.dash_cb_weather["values"] = weathers
            self.dash_cb_weather.set("All")
            self.exp_cb_weather["values"] = weathers
            self.exp_cb_weather.set("All")
        else:
            self.dash_cb_weather["values"] = ["All (N/A)"]
            self.dash_cb_weather.set("All (N/A)")
            self.exp_cb_weather["values"] = ["All (N/A)"]
            self.exp_cb_weather.set("All (N/A)")

        # Road Surface
        if "Road_Surface" in df.columns:
            surfaces = ["All"] + sorted([str(x).title() for x in df["Road_Surface"].dropna().unique() if str(x).strip()])
            self.exp_cb_surface["values"] = surfaces
            self.exp_cb_surface.set("All")
        else:
            self.exp_cb_surface["values"] = ["All (N/A)"]
            self.exp_cb_surface.set("All (N/A)")

        # Populate Geo Aside City Filter
        if hasattr(self, "geo_cb_city"):
            if "City" in df.columns:
                geo_cities = ["All Cities (National View)"] + sorted([str(x) for x in df["City"].dropna().unique() if str(x).strip()])
                self.geo_cb_city["values"] = geo_cities
                self.geo_cb_city.current(0)
            else:
                self.geo_cb_city["values"] = ["All (N/A)"]
                self.geo_cb_city.set("All (N/A)")

    # -------------------------------------------------------------
    # Filter Management
    # -------------------------------------------------------------
    def _on_filter_changed(self, key: str, value: str):
        """Applies active filter change and refreshes views."""
        self.active_filters[key] = value

        # Sync dashboard and explorer dropdowns
        if key == "city":
            self.dash_cb_city.set(value)
            self.exp_cb_city.set(value)
        elif key == "severity":
            self.dash_cb_sev.set(value)
            self.exp_cb_sev.set(value)
        elif key == "weather":
            self.dash_cb_weather.set(value)
            self.exp_cb_weather.set(value)
        elif key == "road_surface":
            self.exp_cb_surface.set(value)

        self._apply_current_filters()

    def _on_search_submitted(self):
        """Handles text search in explorer."""
        query = self.ent_explorer_search.get().strip()
        self.active_filters["search"] = query
        self._apply_current_filters()

    def clear_all_filters(self):
        """Resets all filters to 'All' / empty."""
        self.active_filters = {
            "city": "All",
            "severity": "All",
            "weather": "All",
            "road_surface": "All",
            "search": ""
        }
        self.dash_cb_city.set("All")
        self.dash_cb_sev.set("All")
        self.dash_cb_weather.set("All")
        self.exp_cb_city.set("All")
        self.exp_cb_sev.set("All")
        self.exp_cb_weather.set("All")
        self.exp_cb_surface.set("All")
        self.ent_explorer_search.delete(0, tk.END)

        self._apply_current_filters()

    def _apply_current_filters(self):
        """Filters the working dataframe and updates all dependent views."""
        if self.cleaner.cleaned_df is None or self.cleaner.cleaned_df.empty:
            return

        filtered = self.loader.filter_data(
            df=self.cleaner.cleaned_df,
            city=self.active_filters["city"],
            severity=self.active_filters["severity"],
            weather=self.active_filters["weather"],
            road_surface=self.active_filters["road_surface"],
            search_query=self.active_filters["search"]
        )

        self.current_df = filtered
        self.analyzer.set_data(filtered)

        # Count active filters
        active_count = sum(1 for k, v in self.active_filters.items() if v and v != "All")
        tot = len(self.cleaner.cleaned_df)
        filt_tot = len(filtered)
        self.lbl_side_records.configure(text=f"{filt_tot:,} / {tot:,} records | {active_count} active filters")

        self._update_all_views()

    # -------------------------------------------------------------
    # View Updaters
    # -------------------------------------------------------------
    def _update_all_views(self):
        """Refreshes whichever views are currently visible or cached."""
        self._update_dashboard_kpis()
        self._update_dashboard_charts()
        self._update_explorer_table()
        self._update_quality_view()

        # If user is currently on an analytics, geo, or reports tab, refresh it
        if self.current_view == "analytics":
            self._render_analytics_chart()
        elif self.current_view == "geo":
            self._render_geo_chart()
        elif self.current_view == "reports":
            self._generate_and_display_report()

    def _refresh_view(self, view_key: str):
        """Invoked when navigating to a tab."""
        if view_key == "dashboard":
            self._update_dashboard_kpis()
            self._update_dashboard_charts()
        elif view_key == "explorer":
            self._update_explorer_table()
        elif view_key == "cleaning":
            self._update_quality_view()
        elif view_key == "analytics":
            self._render_analytics_chart()
        elif view_key == "geo":
            self._render_geo_chart()
        elif view_key == "reports":
            self._generate_and_display_report()

    # -------------------------------------------------------------
    # Sub-View Renderers
    # -------------------------------------------------------------
    def _update_dashboard_kpis(self):
        """Recalculates and updates the 6 KPI cards."""
        kpis = self.analyzer.get_kpis()

        self.kpi_widgets["total"][0].configure(text=f"{kpis['total_accidents']:,}")
        self.kpi_widgets["total"][1].configure(text=f"Avg {kpis['avg_vehicles']} veh/crash")

        self.kpi_widgets["fatal"][0].configure(text=f"{kpis['fatal_accidents']:,}")
        self.kpi_widgets["fatal"][1].configure(text=f"{kpis['fatal_pct']}% of total")

        self.kpi_widgets["serious"][0].configure(text=f"{kpis['serious_accidents']:,}")
        self.kpi_widgets["serious"][1].configure(text=f"{kpis['serious_pct']}% of total")

        self.kpi_widgets["minor"][0].configure(text=f"{kpis['minor_accidents']:,}")
        self.kpi_widgets["minor"][1].configure(text=f"{kpis['minor_pct']}% of total")

        self.kpi_widgets["peak_hour"][0].configure(text=str(kpis["peak_hour"]))
        self.kpi_widgets["peak_hour"][1].configure(text=f"Count: {kpis['peak_hour_count']} crashes")

        self.kpi_widgets["locations"][0].configure(text=f"{kpis['unique_locations']:,}")
        self.kpi_widgets["locations"][1].configure(text="Distinct cities")

    def _update_dashboard_charts(self):
        """Draws the Severity Donut and Hourly Line or Geographic Hotspots on the dashboard."""
        # 1. Severity Chart
        if self.dash_fig_sev is None:
            self.dash_fig_sev = Visualizer.plot_severity_pie(self.analyzer)
            self.dash_canvas_sev = FigureCanvasTkAgg(self.dash_fig_sev, master=self.dash_sev_container)
            self.dash_canvas_sev.get_tk_widget().pack(fill="both", expand=True)
        else:
            Visualizer.plot_severity_pie(self.analyzer, fig=self.dash_fig_sev)
            self.dash_canvas_sev.draw()

        # 2. Right Chart: Toggle between Hourly Line and Geographic Scatter
        if getattr(self, "dash_right_mode", "timeline") == "geo":
            if self.dash_fig_time is None:
                self.dash_fig_time = Visualizer.plot_geospatial_hotspots(self.analyzer)
                self.dash_canvas_time = FigureCanvasTkAgg(self.dash_fig_time, master=self.dash_time_container)
                self.dash_canvas_time.get_tk_widget().pack(fill="both", expand=True)
            else:
                Visualizer.plot_geospatial_hotspots(self.analyzer, fig=self.dash_fig_time)
                self.dash_canvas_time.draw()
        else:
            if self.dash_fig_time is None:
                self.dash_fig_time = Visualizer.plot_hourly_trend(self.analyzer)
                self.dash_canvas_time = FigureCanvasTkAgg(self.dash_fig_time, master=self.dash_time_container)
                self.dash_canvas_time.get_tk_widget().pack(fill="both", expand=True)
            else:
                Visualizer.plot_hourly_trend(self.analyzer, fig=self.dash_fig_time)
                self.dash_canvas_time.draw()

    def _toggle_dash_right_chart(self, mode: str):
        """Toggles dashboard right card between Hourly Timeline and Geographic Hotspots."""
        self.dash_right_mode = mode
        if mode == "geo":
            self.lbl_dash_right_title.configure(text="Geographic Accident Hotspots")
            self.btn_dash_geo_toggle.configure(bg=self.COLORS["primary"], fg="#FFFFFF", bd=0)
            self.btn_dash_time_toggle.configure(bg="#F1F5F9", fg=self.COLORS["text_dark"], bd=1)
        else:
            self.lbl_dash_right_title.configure(text="24-Hour Accident Timeline")
            self.btn_dash_time_toggle.configure(bg=self.COLORS["primary"], fg="#FFFFFF", bd=0)
            self.btn_dash_geo_toggle.configure(bg="#F1F5F9", fg=self.COLORS["text_dark"], bd=1)

        self._update_dashboard_charts()

    def _update_explorer_table(self):
        """Populates the Treeview table with current records."""
        # Clear existing
        self.tree_explorer.delete(*self.tree_explorer.get_children())

        df = self.current_df
        if df.empty:
            self.lbl_explorer_status.configure(text="No records match current filter criteria.")
            self.lbl_explorer_meta.configure(text="Columns: 0")
            return

        # Display sensible columns
        display_cols = [c for c in df.columns if c not in ["Parsed_Date"]]
        self.tree_explorer["columns"] = display_cols
        self.tree_explorer["show"] = "headings"

        for col in display_cols:
            self.tree_explorer.heading(
                col, text=col,
                command=lambda c=col: self._sort_treeview_column(c, False)
            )
            # Adjust column width
            max_len = max(df[col].astype(str).str.len().max() if not df[col].empty else 10, len(col))
            col_width = min(max(int(max_len * 9), 75), 180)
            self.tree_explorer.column(col, width=col_width, anchor="center" if "ID" in col or "Severity" in col else "w")

        # Insert rows (up to 1000 for responsive UI)
        for _, row in df.head(1000).iterrows():
            vals = ["" if pd.isna(row[c]) else str(row[c]) for c in display_cols]
            self.tree_explorer.insert("", "end", values=vals)

        tot_matching = len(df)
        self.lbl_explorer_status.configure(text=f"Showing {min(1000, tot_matching):,} of {tot_matching:,} records")
        self.lbl_explorer_meta.configure(text=f"Total Columns: {len(display_cols)}")

    def _sort_treeview_column(self, col: str, reverse: bool):
        """Sorts the Treeview column on header click."""
        try:
            l = [(self.tree_explorer.set(k, col), k) for k in self.tree_explorer.get_children("")]
            # Try numeric sort first
            try:
                l.sort(key=lambda t: float(t[0]), reverse=reverse)
            except ValueError:
                l.sort(key=lambda t: t[0].lower(), reverse=reverse)

            for index, (val, k) in enumerate(l):
                self.tree_explorer.move(k, "", index)

            self.tree_explorer.heading(col, command=lambda: self._sort_treeview_column(col, not reverse))
        except Exception:
            pass

    def _update_quality_view(self):
        """Refreshes quality audit metrics and column completeness table."""
        quality = self.cleaner.assess_quality(self.cleaner.cleaned_df)

        self.clean_metric_labels["total_rows"].configure(text=f"{quality['total_rows']:,}")
        self.clean_metric_labels["duplicates"].configure(text=f"{quality['duplicates_count']:,}")

        # Missing columns count
        miss_count = sum(1 for v in quality["missing_summary"].values() if v["count"] > 0)
        self.clean_metric_labels["missing_cols"].configure(text=str(miss_count))

        self.clean_metric_labels["invalid_coords"].configure(text=f"{quality['invalid_coords_count']:,}")

        # Populate quality table
        self.tree_quality.delete(*self.tree_quality.get_children())
        for col, info in quality["missing_summary"].items():
            self.tree_quality.insert("", "end", values=(
                col,
                info["dtype"],
                f"{info['count']:,}",
                f"{info['percentage']}%",
                f"{info['valid_count']:,}"
            ))

    def _render_analytics_chart(self):
        """Draws the selected in-depth analysis chart in the analytics tab."""
        chart_key = None
        current_name = self.cb_analytics_type.get()
        for k, name in self.analytics_chart_types:
            if name == current_name:
                chart_key = k
                break

        # Dispatch chart generator
        if chart_key == "sev_bar":
            fig = Visualizer.plot_severity_bar(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "sev_pie":
            fig = Visualizer.plot_severity_pie(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "time_hour":
            fig = Visualizer.plot_hourly_trend(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "time_month":
            fig = Visualizer.plot_monthly_trend(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "locations":
            fig = Visualizer.plot_top_locations(self.analyzer, fig=self.analytics_fig, top_n=10)
        elif chart_key == "weather":
            fig = Visualizer.plot_weather_distribution(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "surface":
            fig = Visualizer.plot_road_surface(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "vehicles":
            fig = Visualizer.plot_vehicle_distribution(self.analyzer, fig=self.analytics_fig)
        elif chart_key == "heatmap":
            fig = Visualizer.plot_correlation_heatmap(self.analyzer, fig=self.analytics_fig)
        else:
            fig = Visualizer.plot_severity_bar(self.analyzer, fig=self.analytics_fig)

        self.analytics_fig = fig

        if self.analytics_canvas is None:
            self.analytics_canvas = FigureCanvasTkAgg(self.analytics_fig, master=self.analytics_card)
            self.analytics_canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.analytics_canvas.draw()

    def _render_geo_chart(self):
        """Draws geospatial coordinates scatter plot with aside bar stats."""
        city_sel = getattr(self, "geo_cb_city", None)
        chosen_city = city_sel.get() if city_sel else "All"
        if not chosen_city or "All" in chosen_city:
            city_filter = None
        else:
            city_filter = chosen_city

        sev_sel = getattr(self, "geo_cb_sev", None)
        chosen_sev = sev_sel.get() if sev_sel else "All"
        if not chosen_sev or "All" in chosen_sev:
            sev_filter = None
        else:
            sev_filter = chosen_sev

        fig = Visualizer.plot_geospatial_hotspots(
            self.analyzer,
            fig=self.geo_fig,
            city_filter=city_filter,
            severity_filter=sev_filter
        )
        self.geo_fig = fig

        # Update aside bar spatial statistics
        self._update_geo_aside_metrics(city_filter=city_filter)

        if self.geo_canvas is None:
            self.geo_canvas = FigureCanvasTkAgg(self.geo_fig, master=self.geo_card)
            self.geo_canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.geo_canvas.draw()

    def _on_geo_city_changed(self, event=None):
        """Called when user changes city in geo aside bar."""
        chosen = self.geo_cb_city.get().strip()
        if "All" in chosen:
            self.lbl_geo_header_scope.configure(text="Display: All Municipalities")
        else:
            self.lbl_geo_header_scope.configure(text=f"Display: {chosen}")
        self._render_geo_chart()

    def _on_geo_sev_changed(self, event=None):
        """Called when user changes severity in geo aside bar."""
        self._render_geo_chart()

    def _zoom_to_city(self, city_name: str):
        """Quick action to select city from cluster list and zoom map."""
        for val in self.geo_cb_city["values"]:
            if val.lower() == city_name.lower():
                self.geo_cb_city.set(val)
                break
        else:
            self.geo_cb_city.set(city_name)
        self.lbl_geo_header_scope.configure(text=f"Display: {city_name}")
        self._render_geo_chart()

    def _reset_geo_view(self):
        """Resets the geospatial view to national scope."""
        if hasattr(self, "geo_cb_city") and self.geo_cb_city["values"]:
            self.geo_cb_city.current(0)
        if hasattr(self, "geo_cb_sev"):
            self.geo_cb_sev.set("All Severities")
        if hasattr(self, "lbl_geo_header_scope"):
            self.lbl_geo_header_scope.configure(text="Display: All Municipalities")
        self._render_geo_chart()

    def _export_geo_csv(self):
        """Exports currently plotted GPS records to a CSV file."""
        df = self.analyzer.df
        if df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
            messagebox.showwarning("No Data", "No coordinate data available to export.")
            return

        lat = pd.to_numeric(df["Latitude"], errors="coerce")
        lon = pd.to_numeric(df["Longitude"], errors="coerce")
        valid = (
            lat.notna() & lon.notna() &
            (lat >= -90) & (lat <= 90) &
            (lon >= -180) & (lon <= 180) &
            ~((lat == 0) & (lon == 0))
        )
        valid_df = df[valid].copy()

        chosen_city = self.geo_cb_city.get() if hasattr(self, "geo_cb_city") else "All"
        if chosen_city and "All" not in chosen_city and "City" in valid_df.columns:
            valid_df = valid_df[valid_df["City"].astype(str).str.strip().str.lower() == chosen_city.strip().lower()]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files (*.csv)", "*.csv")],
            initialfile="geospatial_hotspots.csv"
        )
        if file_path:
            success, err = self.loader.export_to_csv(valid_df, file_path)
            if success:
                messagebox.showinfo("Export Successful", f"Saved {len(valid_df):,} GPS records to:\n" + str(file_path))
            else:
                messagebox.showerror("Export Failed", f"Could not save file:\n" + str(err))

    def _update_geo_aside_metrics(self, city_filter=None):
        """Refreshes cards and statistics in the geo aside bar."""
        if not hasattr(self, "lbl_geo_valid_pts"):
            return

        summary = self.analyzer.get_spatial_summary()
        
        # If city filter is active, get stats for that subset
        if city_filter and "All" not in city_filter and "City" in self.analyzer.df.columns:
            sub = self.analyzer.df[self.analyzer.df["City"].astype(str).str.strip().str.lower() == city_filter.strip().lower()]
            sub_analyzer = AccidentAnalyzer(sub)
            summary = sub_analyzer.get_spatial_summary()

        # 1. Coordinate Metrics
        self.lbl_geo_valid_pts.configure(
            text=f"{summary['valid_count']:,} / {summary['total_count']:,} ({summary['valid_pct']}%)"
        )
        self.lbl_geo_lat_bounds.configure(
            text=f"[{summary['lat_min']}° to {summary['lat_max']}° N]"
        )
        self.lbl_geo_lon_bounds.configure(
            text=f"[{summary['lon_min']}° to {summary['lon_max']}° W]"
        )
        self.lbl_geo_centroid.configure(
            text=f"Centroid: ({summary['centroid_lat']}°, {summary['centroid_lon']}°)"
        )

        # 2. Severity counts & percentages
        sev = summary["severity_counts"]
        tot = max(summary["valid_count"], 1)
        self.lbl_geo_sev_fatal.configure(text=f"{sev['Fatal']:,} ({sev['Fatal']/tot*100:.1f}%)")
        self.lbl_geo_sev_serious.configure(text=f"{sev['Serious']:,} ({sev['Serious']/tot*100:.1f}%)")
        self.lbl_geo_sev_minor.configure(text=f"{sev['Minor']:,} ({sev['Minor']/tot*100:.1f}%)")

        # 3. Top Cluster List
        if hasattr(self, "geo_clusters_list_frame"):
            for child in self.geo_clusters_list_frame.winfo_children():
                child.destroy()

            clusters = summary["top_clusters"]
            if not clusters:
                tk.Label(
                    self.geo_clusters_list_frame, text="No clusters available.",
                    bg="#F8FAFC", fg=self.COLORS["text_muted"], font=self.FONTS["small"]
                ).pack(anchor="w")
            else:
                for idx, c in enumerate(clusters, start=1):
                    row_f = tk.Frame(self.geo_clusters_list_frame, bg="#F8FAFC")
                    row_f.pack(fill="x", pady=1)

                    lbl = tk.Label(
                        row_f, text=f"#{idx} {c['city']} ({c['count']})",
                        bg="#F8FAFC", fg=self.COLORS["text_dark"], font=self.FONTS["small"]
                    )
                    lbl.pack(side="left")

                    btn = tk.Button(
                        row_f, text="Zoom", bg="#FFFFFF", fg=self.COLORS["primary"],
                        font=("Segoe UI", 7, "bold"), bd=1, relief="solid", padx=4, pady=0, cursor="hand2",
                        command=lambda city=c['city']: self._zoom_to_city(city)
                    )
                    btn.pack(side="right")

    def _generate_and_display_report(self):
        """Generates formatted academic report string and displays in preview widget."""
        fname = self.loader.file_path or "Dataset"
        report_text = ReportGenerator.generate_text_report(
            self.analyzer,
            dataset_name=fname,
            active_filters=self.active_filters
        )
        self.txt_report_preview.delete("1.0", tk.END)
        self.txt_report_preview.insert("1.0", report_text)

    # -------------------------------------------------------------
    # Cleaning Action Handlers
    # -------------------------------------------------------------
    def _execute_cleaning(self):
        """Runs user-selected cleaning operations."""
        if self.cleaner.original_df is None or self.cleaner.original_df.empty:
            messagebox.showwarning("No Dataset", "Please load a dataset before executing cleaning operations.")
            return

        cleaned_df, report = self.cleaner.clean(
            drop_duplicates=self.var_drop_dups.get(),
            fill_missing_categorical=self.var_fill_cat.get(),
            fill_missing_numeric=self.var_fill_num.get(),
            normalize_severities=self.var_norm_sev.get(),
            sanitize_coordinates=self.var_sanitize_coords.get()
        )

        # Update log box
        self.txt_clean_log.delete("1.0", tk.END)
        for entry in report.get("audit_log", []):
            self.txt_clean_log.insert(tk.END, f"✔ {entry}\n")

        # Update working dataset and refresh views
        self.current_df = cleaned_df
        self.analyzer.set_data(cleaned_df)
        self._populate_dropdowns(cleaned_df)
        self._update_all_views()

        m = report.get("metrics", {})
        messagebox.showinfo(
            "Cleaning Complete",
            f"Dataset Cleaning Finished:\n"
            f"• Initial Rows: {m.get('initial_rows', 0):,}\n"
            f"• Duplicates Removed: {m.get('duplicates_removed', 0):,}\n"
            f"• Missing Fields Imputed: {m.get('missing_filled', 0):,}\n"
            f"• Final Valid Rows: {m.get('final_rows', 0):,}"
        )

    def _restore_original_data(self):
        """Restores original uncleaned dataset."""
        if self.cleaner.original_df is None:
            return

        restored = self.cleaner.restore_original()
        self.current_df = restored
        self.analyzer.set_data(restored)

        self.txt_clean_log.delete("1.0", tk.END)
        self.txt_clean_log.insert(tk.END, "✔ Dataset successfully restored to original imported state.\n")

        self._populate_dropdowns(restored)
        self._update_all_views()
        messagebox.showinfo("Restored", "Original dataset has been successfully restored!")

    # -------------------------------------------------------------
    # Export Handlers
    # -------------------------------------------------------------
    def _export_filtered_dataset(self):
        """Exports currently filtered records to a CSV file."""
        if self.current_df.empty:
            messagebox.showwarning("Empty Data", "There are no records to export under current filters.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files (*.csv)", "*.csv")],
            initialfile="filtered_accidents.csv"
        )
        if file_path:
            success, err = self.loader.export_to_csv(self.current_df, file_path)
            if success:
                messagebox.showinfo("Export Successful", f"Saved {len(self.current_df):,} records to:\n{file_path}")
            else:
                messagebox.showerror("Export Failed", f"Could not save file:\n" + str(err))

    def _export_cleaned_dataset(self):
        """Exports cleaned dataset to a CSV file."""
        if self.cleaner.cleaned_df is None or self.cleaner.cleaned_df.empty:
            messagebox.showwarning("No Data", "No cleaned dataset available to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files (*.csv)", "*.csv")],
            initialfile="cleaned_accidents.csv"
        )
        if file_path:
            success, err = self.loader.export_to_csv(self.cleaner.cleaned_df, file_path)
            if success:
                messagebox.showinfo("Export Successful", f"Cleaned dataset saved successfully to:\n{file_path}")
            else:
                messagebox.showerror("Export Failed", f"Could not save file:\n{err}")

    def _export_report_txt(self):
        """Exports current report to text file."""
        content = self.txt_report_preview.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("No Report", "Please generate a report first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files (*.txt)", "*.txt"), ("Markdown (*.md)", "*.md")],
            initialfile="Accident_Analytics_Report.txt"
        )
        if file_path:
            success, err = ReportGenerator.export_text_report(content, file_path)
            if success:
                messagebox.showinfo("Report Saved", f"Report saved successfully to:\n{file_path}")
            else:
                messagebox.showerror("Save Failed", f"Error saving report:\n{err}")

    def _export_summary_tables(self):
        """Exports summary CSV tables to chosen directory."""
        dir_path = filedialog.askdirectory(title="Select Folder to Export Summary CSV Tables")
        if dir_path:
            success, files, err = ReportGenerator.export_summary_tables(self.analyzer, dir_path)
            if success:
                file_list_str = "\n".join([f"• {os.path.basename(f)}" for f in files])
                messagebox.showinfo("Tables Exported", f"Successfully exported {len(files)} summary tables:\n{file_list_str}")
            else:
                messagebox.showerror("Export Failed", f"Could not export tables:\n{err}")

    def _save_chart_as_image(self, fig: Optional[Figure], default_filename: str):
        """Saves a Matplotlib figure to a high-DPI image."""
        if fig is None:
            messagebox.showwarning("No Chart", "No active chart to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image (*.png)", "*.png"), ("PDF Document (*.pdf)", "*.pdf"), ("SVG Image (*.svg)", "*.svg")],
            initialfile=default_filename
        )
        if file_path:
            success, err = Visualizer.save_figure(fig, file_path, dpi=300)
            if success:
                messagebox.showinfo("Chart Saved", f"High-resolution chart saved to:\n{file_path}")
            else:
                messagebox.showerror("Save Failed", f"Could not save chart:\n{err}")
