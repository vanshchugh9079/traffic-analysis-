"""
Traffic Accident Analyzer - Main Application Entry Point
A modern college-level Python data analytics desktop application using Tkinter,
Pandas, NumPy, and Matplotlib.
"""
import os
import sys

# Ensure current and parent directories are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Set Windows High DPI Awareness for crisp fonts
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from modules.dashboard import ModernDashboard


def main():
    """Launches the Traffic Accident Analyzer desktop GUI application."""
    app = ModernDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
